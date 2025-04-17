# mode_switch_synth.py
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100

# Define modes as semitone intervals from root
MODE_ORDER = [
    'locrian', 'phrygian', 'aeolian', 'lydian', 'dorian', 'mixolydian', 'ionian'
]
MODES = {
    'ionian':     [0, 2, 4, 5, 7, 9, 11],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'phrygian':   [0, 1, 3, 5, 7, 8, 10],
    'lydian':     [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian':    [0, 2, 3, 5, 7, 8, 10],
    'locrian':    [0, 1, 3, 5, 6, 8, 10]
}

BASE_MIDI = 60
pattern = []
pattern_index = 0
pattern_root = BASE_MIDI
notes_played = 0


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def generate_sine(freq, duration, amp):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    wave = amp * np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not found")
    exit()

print("[INFO] Playing structured random notes. Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Extract warmth
    small = cv2.resize(frame, (160, 120))
    r_mean = np.mean(small[:, :, 2])
    b_mean = np.mean(small[:, :, 0])
    warmth = r_mean / (b_mean + 1.0)
    norm_warmth = min(warmth / 3.0, 1.0)

    # Choose mode based on warmth
    mode_index = int(norm_warmth * len(MODE_ORDER)) % len(MODE_ORDER)
    mode_name = MODE_ORDER[mode_index]
    mode = MODES[mode_name]

    # Generate new pattern every 4 notes
    if notes_played % 4 == 0:
        pattern_root = BASE_MIDI + 12 * random.randint(0, 1)
        pattern = [random.choice(mode) for _ in range(4)]
        pattern_index = 0

    # Select interval from pattern and create transposed note
    pattern_offset = (notes_played // 4) % 4 * 2  # Shift every 4 notes
    note_midi = pattern_root + pattern[pattern_index] + pattern_offset
    pattern_index = (pattern_index + 1) % 4

    # Duration variation
    duration_beats = random.randint(1, 4)
    duration_sec = 0.5 * duration_beats

    freq = midi_to_freq(note_midi)
    print(f"[MODE: {mode_name}] Note MIDI: {note_midi}, Freq: {freq:.2f} Hz, Duration: {duration_sec:.1f}s")

    tone = generate_sine(freq, duration_sec, 0.4)
    sd.play(tone, samplerate=fs)
    sd.wait()

    notes_played += 1

    # Show camera
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()