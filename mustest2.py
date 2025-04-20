# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import threading
import random
import time

fs = 44100

# 10 base chords: mix of minor7, major7, and addsuspended voices
CHORDS = [
    # Darker: using minor7 and suspended chords
    [69, 76, 79, 69+12, 76+12, 79+12],  # A sus2 (A,B,E) with octave repeats
    [62, 65, 69, 72, 62+12, 69+12],      # D minor7
    [63, 68, 70, 63+12, 68+12, 70+12],   # Eb sus4 (Eb,Ab,Bb)
    [65, 72, 75, 65+12, 72+12, 75+12],    # F minor7
    [67, 69, 74, 67+12, 69+12, 74+12],    # G sus2 (G,A,D)
    # Brighter: major7 and suspended
    [60, 64, 67, 71, 60+12, 67+12],      # C major7
    [62, 66, 69, 62+12, 66+12, 69+12],    # D sus4 (D,G,A)
    [64, 68, 71, 75, 64+12, 71+12],      # E major7
    [65, 67, 72, 65+12, 67+12, 72+12],    # F sus2 (F,G,C)
    [67, 71, 74, 78, 67+12, 74+12]       # G major7
]  # A minor7

MODES = ['rev_arp', 'random', 'forward_arp']

duration = 4.0  # full bar duration (3/4 time)
sub = duration / 6.0  # subdivision for 6-note arp

# Shared state
global_seq = []
global_waveform = 'sine'
global_texture = 0.0
running = True
seq_lock = threading.Lock()
wav_lock = threading.Lock()

# Utility lambdas
midi_to_freq = lambda m: 440.0 * 2 ** ((m - 69) / 12)
remap = lambda v,a,b,c,d: (v-a)/(b-a)*(d-c)+c

# Wave generator
def generate_wave(freq, length, amp, kind):
    t = np.linspace(0, length, int(fs*length), endpoint=False)
    if kind == 'sine': return amp * np.sin(2*np.pi*freq*t)
    if kind == 'triangle': return amp * (2*np.abs(2*((t*freq)%1)-1)-1)
    if kind == 'square': return amp * np.sign(np.sin(2*np.pi*freq*t))
    return np.zeros_like(t)

# Video capture
cap = cv2.VideoCapture(0)

def get_features():
    ret, frame = cap.read()
    if not ret: return None
    small = cv2.resize(frame, (160,120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0
    r, g, b = [np.mean(small[:,:,i]) for i in (2,1,0)]
    warmth = remap((2*r)/(2*(b+g)+1), 0.45, 0.55, 0, len(CHORDS)-0.001)
    edges = cv2.Canny(gray, 50, 150)
    texture = np.count_nonzero(edges) / edges.size
    print(f"[FEATURES] brightness={brightness:.2f}, warmth={warmth:.2f}, texture={texture:.3f}")
    return frame, warmth, brightness, texture

# Chord updater: runs every full bar
def chord_updater():
    global global_seq
    while running:
        feat = get_features()
        if feat is None:
            time.sleep(0.1)
            continue
        _, warmth, brightness, _ = feat
        chord = CHORDS[min(int(warmth), len(CHORDS)-1)]
        mode = random.choice(MODES)
        seq = chord.copy()
        if mode == 'rev_arp': seq.reverse()
        elif mode == 'random': random.shuffle(seq)
        # transpose by octaves based on brightness
        octave_offset = int(remap(brightness, 0.0, 1.0, -2, 2)) * 12
        seq = [n + octave_offset for n in seq]
        seq.append(seq[0])  # repeat first for 6th note
        with seq_lock:
            global_seq = seq
        time.sleep(duration)

# Waveform & texture updater: runs every subdivision
def waveform_updater():
    global global_waveform, global_texture
    while running:
        feat = get_features()
        if feat is None:
            time.sleep(0.05)
            continue
        _, _, _, texture = feat
        wf = 'sine' if texture < 0.045 else ('triangle' if texture < 0.55 else 'square')
        with wav_lock:
            global_waveform = wf
            global_texture = texture
        time.sleep(sub)

# Player: plays one note per subdivision with release envelope based on texture
def player():
    idx = 0
    while running:
        with seq_lock:
            seq = list(global_seq)
        with wav_lock:
            wf = global_waveform
            texture = global_texture
        if seq:
            note = seq[idx % len(seq)]
            freq = midi_to_freq(note)
            wave = generate_wave(freq, sub, sub, wf)
            print(f"[PLAYER] note={note}, freq={freq:.2f}Hz, waveform={wf}, texture={texture:.3f}")
            # Release-only envelope: super-sensitive to texture
            rel_len = int((1 - texture**2) * len(wave))
            if rel_len > 0:
                env = np.ones_like(wave)
                env[-rel_len:] = np.linspace(1, 0, rel_len)
                wave *= env
            sd.play(wave / np.max(np.abs(wave)), fs)
            sd.wait()
            idx += 1
        time.sleep(0)

# Start threads
t1 = threading.Thread(target=chord_updater, daemon=True)
t2 = threading.Thread(target=waveform_updater, daemon=True)
t3 = threading.Thread(target=player, daemon=True)
t1.start(); t2.start(); t3.start()

print("[INFO] Multi-threaded ARP synth. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret: continue
    cv2.imshow('Cam', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break

cap.release()
cv2.destroyAllWindows()
