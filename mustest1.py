# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100

# 10 base chords
# CHORDS = [
#     [69, 71, 76, 78, 82, 70],  # A dark    (add Bb = b9 for tension)
#     [61, 64, 67, 70, 74, 60],  # B dim     (add C  = dim7 extension)
#     [62, 65, 69, 86, 88, 63],  # Dm        (add Eb = m9)
#     [63, 66, 70, 72, 75, 64],  # Ebm       (add Fb = E natural as b9)
#     [67, 70, 74, 76, 77, 68],  # G7        (add Ab = b9 tension)
#     [64, 68, 71, 73, 76, 75],  # Emaj      (add D# = maj7 warmth)
#     [65, 69, 72, 74, 81, 79],  # Fmaj      (add G  = 9th for brightness)
#     [66, 70, 73, 75, 78, 80],  # F#maj7    (add A# = 9th lift)
#     [60, 64, 67, 71, 74, 76],  # Cmaj      (add E  = maj7 shimmer)
#     [59, 62, 66, 69, 72, 74]   # Bmaj      (add D# = 9th color)
# ]
CHORDS = [
    [69, 72, 76, 79, 70, 75],  # A dark: Am7(b5,b9)
    [61, 64, 67, 68, 60, 63],  # B dim: fully diminished 7th
    [62, 65, 69, 74, 76, 71],  # Dm: Dm9
    [63, 66, 70, 73, 75, 68],  # Ebm: Ebm9(b9)
    [67, 71, 74, 79, 69, 72],  # G7: G9 with 7
    [64, 68, 71, 75, 73, 78],  # Emaj: Emaj7(9)
    [65, 69, 72, 76, 79, 62],  # Fmaj: Fmaj9
    [66, 70, 73, 77, 81, 71],  # F#maj7: F#maj9
    [60, 64, 67, 71, 62, 69],  # Cmaj: Cmaj9
    [59, 63, 66, 70, 61, 68]   # Bmaj: Bmaj9
]
MODES = ['rev_arp', 'random', 'forward_arp']

# Helpers
midi_to_freq = lambda m: 440.0 * 2 ** ((m - 69) / 12)
remap = lambda v,a,b,c,d: (v-a)/(b-a)*(d-c)+c
# Total bar duration (3/4): 6 subdivisions (6 notes)
duration = 3

# Wave generator
def generate_wave(freq, length, amp, kind):
    t = np.linspace(0, length, int(fs*length), endpoint=False)
    if kind == 'sine':
        return amp * np.sin(2*np.pi*freq*t)
    if kind == 'triangle':
        return amp * (2*np.abs(2*((t*freq)%1)-1)-1)
    if kind == 'square':
        return amp * np.sign(np.sin(2*np.pi*freq*t))
    return np.zeros_like(t)

# Extract visual features
def get_visual_features(frame):
    small = cv2.resize(frame, (160,120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)/255.0
    r,g,b = [np.mean(small[:,:,i]) for i in (2,1,0)]
    warmth = remap((2*r)/(2*(b+g)+1), 0.45,0.55, 0, len(CHORDS)-0.001)
    edges = cv2.Canny(gray,50,150)
    texture = np.count_nonzero(edges)/edges.size
    return warmth, brightness, texture

# Select chord by warmth
def select_chord(warmth):
    idx = min(int(np.floor(warmth)), len(CHORDS)-1)
    return CHORDS[idx]

# Select waveform by texture
def select_waveform(texture):
    return 'sine' if texture<0.07 else ('triangle' if texture<0.14 else 'square')

# Main ARP synth
cap = cv2.VideoCapture(0)
print("[INFO] ARP synth (6-note bar). Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret: continue

    warmth, brightness, texture = get_visual_features(frame)
    chord = select_chord(warmth)
    mode = random.choice(MODES)
    waveform = select_waveform(texture)

    # Build 6-note sequence: 5 chord notes + repeat first for continuity
    seq = chord.copy()
    if mode == 'rev_arp':
        seq.reverse()
    elif mode == 'random':
        random.shuffle(seq)
    # Transpose by brightness (-2 to +2 semitones)
    offset_oct = int(remap(brightness, 0.0, 1.0, -2, 2))
    seq = [n + offset_oct * 12 for n in seq]

    # Subdivision length
    sub = duration / 6.0

    # Play all 6 notes
    for note in seq:
        freq = midi_to_freq(note)
        wave = generate_wave(freq, sub, sub, waveform)
        # Release-only envelope based on texture
        rel_len = int((1 - texture**1.1) * len(wave))
        if rel_len > 0:
            env = np.ones_like(wave)
            env[-rel_len:] = np.linspace(1, 0, rel_len)
            wave *= env
        sd.play(wave / np.max(np.abs(wave)), fs)
        sd.wait()

    # Loop immediately (no rest)
    cv2.imshow('Cam', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
