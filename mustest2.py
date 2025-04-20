# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import threading
import random
import time

fs = 44100

# 10 base chords
CHORDS = [
    # Dark (minor7 chords)
    [69, 72, 76, 79, 69+12, 76+12],  # A m7: A, C, E, G, A', E'
    [62, 65, 69, 72, 62+12, 69+12],  # D m7: D, F, A, C, D', A'
    [63, 66, 70, 73, 63+12, 70+12],  # Eb m7: Eb, Gb, Bb, Db, Eb', Bb'
    [65, 68, 72, 75, 65+12, 72+12],  # F m7: F, Ab, C, Eb, F', C'
    [67, 70, 74, 77, 67+12, 74+12],  # G m7: G, Bb, D, F, G', D'
    # Happy (major7 chords)
    [60, 64, 67, 71, 60+12, 67+12],  # C maj7: C, E, G, B, C', G'
    [62, 66, 69, 73, 62+12, 69+12],  # D maj7: D, F#, A, C#, D', A'
    [64, 68, 71, 75, 64+12, 71+12],  # E maj7: E, G#, B, D#, E', B'
    [65, 69, 72, 76, 65+12, 72+12],  # F maj7: F, A, C, E, F', C'
    [67, 71, 74, 78, 67+12, 74+12]   # G maj7: G, B, D, F#, G', D'
]
MODES = ['rev_arp', 'random', 'forward_arp']

duration = 3  # full bar duration
sub = duration / 6.0  # subdivision for 6-note arpeggio

# Shared state
global_seq = []
global_waveform = 'sine'
running = True
seq_lock = threading.Lock()
wave_lock = threading.Lock()

# Utility lambdas
midi_to_freq = lambda m: 440.0 * 2 ** ((m - 69) / 12)
remap = lambda v,a,b,c,d: (v-a)/(b-a)*(d-c)+c

# Wave generator
def generate_wave(freq, length, amp, kind):
    t = np.linspace(0, length, int(fs * length), endpoint=False)
    if kind == 'sine':
        return amp * np.sin(2 * np.pi * freq * t)
    elif kind == 'triangle':
        return amp * (2 * np.abs(2 * ((t * freq) % 1) - 1) - 1)
    elif kind == 'square':
        return amp * np.sign(np.sin(2 * np.pi * freq * t))
    else:
        return np.zeros_like(t)

# Feature extractor
cap = cv2.VideoCapture(0)
def get_features():
    ret, frame = cap.read()
    if not ret:
        return None
    small = cv2.resize(frame, (160,120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)/255.0
    r,g,b = [np.mean(small[:,:,i]) for i in (2,1,0)]
    warmth = remap((2*r)/(2*(b+g)+1), 0.45,0.55, 0, len(CHORDS)-0.001)
    edges = cv2.Canny(gray,50,150)
    texture = np.count_nonzero(edges)/edges.size
    return frame, warmth, brightness, texture

# Chord updater: runs every full bar
def chord_updater():
    global global_seq
    while running:
        result = get_features()
        if result is None:
            time.sleep(0.1)
            continue
        frame, warmth, brightness, _ = result
        chord = CHORDS[min(int(warmth), len(CHORDS)-1)]
        mode = random.choice(MODES)
        seq = chord.copy()
        if mode == 'rev_arp': seq.reverse()
        elif mode == 'random': random.shuffle(seq)
        # transpose by octaves based on brightness
        offs = int(remap(brightness, 0.0,1.0, -2,2)) * 12
        seq = [n+offs for n in seq]
        with seq_lock:
            global_seq = seq
        time.sleep(duration)

# Waveform updater: runs every subdivision
def waveform_updater():
    global global_waveform
    while running:
        result = get_features()
        if result is None:
            time.sleep(0.05)
            continue
        _, _, _, texture = result
        wf = 'sine' if texture<0.065 else ('triangle' if texture<0.12 else 'square')
        with wave_lock:
            global_waveform = wf
        time.sleep(sub)

# Player: plays note every subdivision
def player():
    idx = 0
    while running:
        with seq_lock:
            seq = list(global_seq)
        with wave_lock:
            wf = global_waveform
        if seq:
            note = seq[idx % len(seq)]
            freq = midi_to_freq(note)
            wave = generate_wave(freq, sub, sub, wf)
            sd.play(wave/np.max(np.abs(wave)), fs)
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
