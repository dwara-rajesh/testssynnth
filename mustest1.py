# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100

# Define 5 base chords (triads) as MIDI intervals
CHORDS = [
    [60, 64, 67, 71, 74],   # C major
    [62, 65, 69, 86, 88],   # D minor
    [65, 69, 72,  74, 81],   # F major
    [67, 70, 74, 76, 77],   # G7
    [69, 71, 76, 78, 82]    # A somthing
]

# Playback modes
MODES = ['poly', 'arp', 'both']


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def generate_sine(freq, duration, amp):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    wave = amp * np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)

def remap(value, in_min, in_max, out_min, out_max):
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def get_warmth_color(frame):
    resized = cv2.resize(frame, (160, 120))
    r = np.mean(resized[:, :, 2])
    g = np.mean(resized[:, :, 1])
    b = np.mean(resized[:, :, 0])
    warmth_ratio = (2 * r) / (2 * b + g + 1.0)  # Include green in both numerator and denominator
    norm_warmth = remap(warmth_ratio, 0.3, 1.2, 0.0, 1.0)  # Normalized remap with adjusted range
    return norm_warmth


def select_chord_by_color(warmth):
    index =  int(warmth * len(CHORDS)) % len(CHORDS)
    print(index)
    return CHORDS[index]

def play_chord(chord, mode, duration):
    if mode == 'poly':
        waves = [generate_sine(midi_to_freq(n), duration, 0.3) for n in chord]
        buffer = sum(np.pad(w, (0, max(0, len(waves[0]) - len(w)))) for w in waves)
        sd.play(buffer / max(abs(buffer)), samplerate=fs)
        sd.wait()

    elif mode == 'arp':
        for n in chord:
            wave = generate_sine(midi_to_freq(n), duration / 3, (duration/4))
            sd.play(wave, samplerate=fs)
            sd.wait()

    elif mode == 'both':
        waves = []
        for i, n in enumerate(chord):
            wave = generate_sine(midi_to_freq(n), duration, (duration/len(chord)))
            delay = int(i * duration/len(chord) * fs)
            wave = np.pad(wave, (delay, 0))[:int(fs * duration)]
            waves.append(wave)
        buffer = sum(np.pad(w, (0, max(0, len(waves[0]) - len(w)))) for w in waves)
        sd.play(buffer / max(abs(buffer)), samplerate=fs)
        sd.wait()


cap = cv2.VideoCapture(0)
print("[INFO] Playing base chords from webcam input. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    warmth = get_warmth_color(frame)
    chord = select_chord_by_color(warmth)
    mode = 'both' #random.choice(MODES)
    duration = 2.5

    print(f"Warmth: {warmth:.2f} | Mode: {mode} | Chord: {chord}")
    play_chord(chord, mode, duration)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
