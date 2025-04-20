# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100

# 10 base chords
CHORDS = [
    [69, 71, 76, 78, 82],  # A dark
    [61, 64, 67, 70, 74],  # B dim
    [62, 65, 69, 86, 88],  # Dm
    [63, 66, 70, 72, 75],  # Ebm
    [67, 70, 74, 76, 77],  # G7
    [64, 68, 71, 73, 76],  # Emaj
    [65, 69, 72, 74, 81],  # Fmaj
    [66, 70, 73, 75, 78],  # F#maj7
    [60, 64, 67, 71, 74],  # Cmaj
    [59, 62, 66, 69, 72]   # Bmaj
]
MODES = ['rev_arp', 'random', 'forward_arp']

# Helpers
duration = 4.5
midi_to_freq = lambda m: 440.0 * 2 ** ((m - 69) / 12)
remap = lambda v,a,b,c,d: (v-a)/(b-a)*(d-c)+c

def generate_wave(freq, length, amp, kind):
    t = np.linspace(0, length, int(fs*length), endpoint=False)
    if kind == 'sine':
        return amp * np.sin(2*np.pi*freq*t)
    if kind == 'triangle':
        return amp * (2*np.abs(2*((t*freq)%1)-1)-1)
    if kind == 'square':
        return amp * np.sign(np.sin(2*np.pi*freq*t))
    return np.zeros_like(t)


def get_visual_features(frame):
    small = cv2.resize(frame, (160,120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)/255.0
    r,g,b = [np.mean(small[:,:,i]) for i in (2,1,0)]
    warmth = remap((2*r)/(2*(b+g)+1), 0.45,0.55, 0, len(CHORDS)-0.001)
    edges = cv2.Canny(gray,50,150)
    texture = np.count_nonzero(edges)/edges.size
    return warmth, brightness, texture


def select_chord(warmth):
    idx = min(int(np.floor(warmth)), len(CHORDS)-1)
    return CHORDS[idx]


def select_waveform(texture):
    print(texture)
    return 'sine' if texture<0.07 else ('triangle' if texture<0.15 else 'square')

# Main ARP loop
cap = cv2.VideoCapture(0)
print("[INFO] ARP synth. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret: continue

    warmth, brightness, texture = get_visual_features(frame)
    chord = select_chord(warmth)
    mode = random.choice(MODES)
    waveform = select_waveform(texture)

    # 3/4 feel: 6 subdivisions (5 notes+rest)
    sub = duration/6.0
    seq = chord.copy()
    if mode == 'rev_arp': seq.reverse()
    elif mode == 'random': random.shuffle(seq)

    # Transpose range based on brightness
    offset = int(remap(brightness, 0,1, -2,2))
    seq = [n+offset for n in seq]

    # Play 5 notes then rest
    for note in seq:
        freq = midi_to_freq(note)
        wave = generate_wave(freq, sub, sub, waveform)
        # Release-only envelope based on texture: super-sensitive mapping
        # higher texture → shorter release; lower texture → longer release
        rel_len = int((1 - texture**2) * len(wave))
        if rel_len > 0:
            env = np.ones_like(wave)
            env[-rel_len:] = np.linspace(1, 0, rel_len)
            wave *= env
        sd.play(wave/np.max(np.abs(wave)), fs)
        sd.wait()
    time.sleep(sub)  # rest

    cv2.imshow('Cam', frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()
