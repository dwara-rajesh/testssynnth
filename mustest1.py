# mode_switch_synth.py
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100

# Define modes as semitone intervals from root
MODES = {
    'ionian':     [0, 2, 4, 5, 7, 9, 11],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'phrygian':   [0, 1, 3, 5, 7, 8, 10],
    'lydian':     [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian':    [0, 2, 3, 5, 7, 8, 10],
    'locrian':    [0, 1, 3, 5, 6, 8, 10]
}
mode_names = list(MODES.keys())

# Base MIDI note (C4)
BASE_MIDI = 60

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

print("[INFO] Playing random notes. Press ESC to stop.")

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
    mode_index = int(norm_warmth * len(mode_names)) % len(mode_names)
    mode = MODES[mode_names[mode_index]]

    # Random note from current mode
    root_midi = BASE_MIDI + 12 * random.randint(0, 1)  # C4 or C5
    interval = random.choice(mode)
    note_midi = root_midi + interval
    freq = midi_to_freq(note_midi)

    # Generate and play
    print(f"[MODE: {mode_names[mode_index]}] Note MIDI: {note_midi}, Freq: {freq:.2f} Hz")
    tone = generate_sine(freq, 0.8, 0.4)
    sd.play(tone, samplerate=fs)
    sd.wait()

    # Show camera
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
