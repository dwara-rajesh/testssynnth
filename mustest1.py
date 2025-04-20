# chord_synth.py (stripped-down arp generator)
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100  # sample rate

# Define 10 base chords (5 original + 5 inserted)
CHORDS = [
    [69, 71, 76, 78, 82],  # A variant
    [61, 64, 67, 70, 74],  # B dim
    [62, 65, 69, 86, 88],  # D minor
    [63, 66, 70, 72, 75],  # Eb minor
    [67, 70, 74, 76, 77],  # G7
    [64, 68, 71, 73, 76],  # E major
    [65, 69, 72, 74, 81],  # F major
    [66, 70, 73, 75, 78],  # F#maj7
    [60, 64, 67, 71, 74],  # C major
    [59, 62, 66, 69, 72],  # B major
]

MODES = ['rev_arp', 'random', 'forward_arp', 'random']

# Map MIDI note to frequency
def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)

# Generate a single sine tone (mono)
def generate_tone(freq, duration, amp=0.2):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

# Simple linear remap
def remap(v, a, b, c, d):
    return (v - a) / (b - a) * (d - c) + c

# Extract visual features: brightness, warmth, texture
def get_visual(frame):
    small = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0
    r, g, b = [np.mean(small[:,:,i]) for i in (2,1,0)]
    warmth_ratio = (2*r) / (2*(b+g)+1)
    warmth = remap(warmth_ratio, 0.45, 0.55, 0.0, len(CHORDS)-1)
    edges = cv2.Canny(gray,50,150)
    texture = np.count_nonzero(edges) / edges.size
    return brightness, warmth, texture

# Choose chord index by warmth
def select_chord(w):
    idx = int(np.clip(np.floor(w), 0, len(CHORDS)-1))
    return CHORDS[idx]

# Choose mode
def select_mode():
    return random.choice(MODES)

# Arpeggiate chord in 3/4: 5 notes + 1 beat rest
# Duration is total measure length in seconds

def play_arpeggio(chord, mode, duration):
    beat = duration / 3.0       # quarter note = 1 beat
    slot = beat * 0.75          # 5 notes over 3 beats: each gets 0.75 beat
    notes = chord.copy()
    if mode == 'rev_arp':
        notes = list(reversed(notes))
    elif mode == 'random':
        random.shuffle(notes)
    # forward_arp leaves order

    # Play 5 notes
    for note in notes:
        freq = midi_to_freq(note)
        tone = generate_tone(freq, slot)
        sd.play(tone, fs)
        sd.wait()
    # rest for one slot (6th)
    time.sleep(slot)

# Main loop
def main():
    cap = cv2.VideoCapture(0)
    print("[INFO] 3/4 arp generator active. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        b, w, t = get_visual(frame)
        chord = select_chord(w)
        mode = select_mode()
        # duration stays mapping from brightness
        if b >= 0.75:
            duration = 2.0
        elif b >= 0.45:
            duration = 2.5
        else:
            duration = 3.0

        print(f"Bright: {b:.2f} Warmth: {w:.2f} Texture: {t:.3f} Mode: {mode} Chord: {chord}")
        play_arpeggio(chord, mode, duration)

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
