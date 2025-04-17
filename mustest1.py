# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import random
import time

fs = 44100

# Define 10 base chords (5 original + 5 inserted for dynamic range)
CHORDS = [
    [69, 71, 76, 78, 82],     # A something (dark)
    [61, 64, 67, 70, 74],     # B diminished feel (dark and unstable)
    [62, 65, 69, 86, 88],     # D minor
    [63, 66, 70, 72, 75],     # Eb minor with flavor
    [67, 70, 74, 76, 77],     # G7
    [64, 68, 71, 73, 76],     # E major with color
    [65, 69, 72, 74, 81],     # F major
    [66, 70, 73, 75, 78],     # F#maj7 sharp
    [60, 64, 67, 71, 74],     # C major (bright)
    [59, 62, 66, 69, 72]      # B major (bright, colorful)
]

# Playback modes
MODES = ['rev_arp', 'random', 'forward_arp', 'random']

WAVEFORMS = ['sine', 'square', 'triangle', 'noise']

def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)

def generate_wave(freq, duration, amp, kind):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if kind == 'sine':
        wave = amp * np.sin(2 * np.pi * freq * t)
    elif kind == 'square':
        wave = amp * np.sign(np.sin(2 * np.pi * freq * t))
    elif kind == 'triangle':
        wave = amp * 2 * np.abs(2 * ((t * freq) % 1) - 1) - 1
    elif kind == 'noise':
        wave = amp * np.random.uniform(-1, 1, size=t.shape)
    else:
        wave = np.zeros_like(t)
    return wave.astype(np.float32)

def remap(value, in_min, in_max, out_min, out_max):
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

def get_visual_features(frame):
    resized = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    r = np.mean(resized[:, :, 2])
    g = np.mean(resized[:, :, 1])
    b = np.mean(resized[:, :, 0])

    warmth_ratio = (2 * r) / (2 * (b + g) + 1)
    norm_warmth = remap(warmth_ratio, 0.45, 0.55, 0.4, 9.9)

    brightness = np.mean(gray) / 255.0

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    return norm_warmth, brightness, edge_density

def select_chord_by_color(warmth):
    index = min(int(np.floor(warmth)), len(CHORDS) - 1)
    return CHORDS[index]

def select_waveform(texture):
    if texture < 0.04:
        return 'sine'
    elif texture < 0.054:
        return 'triangle'
    elif texture < 0.62:
        return 'square'
    else:
        return 'noise'

def play_chord(chord, mode, duration, waveform):
    if mode == 'rev_arp':
        chordrev = list(reversed(chord))
        waves = []
        for i, n in enumerate(chordrev):
            wave = generate_wave(midi_to_freq(n), duration, (duration/len(chord)), waveform)
            delay = int(i * duration/len(chord) * fs)
            wave = np.pad(wave, (delay, 0))[:int(fs * duration)]
            waves.append(wave)
        buffer = sum(np.pad(w, (0, max(0, len(waves[0]) - len(w)))) for w in waves)
        sd.play(buffer / max(abs(buffer)), samplerate=fs)
        sd.wait()

    elif mode == 'forward_arp':
        waves = []
        for i, n in enumerate(chord):
            wave = generate_wave(midi_to_freq(n), duration, (duration/len(chord)), waveform)
            delay = int(i * duration/len(chord) * fs)
            wave = np.pad(wave, (delay, 0))[:int(fs * duration)]
            waves.append(wave)
        buffer = sum(np.pad(w, (0, max(0, len(waves[0]) - len(w)))) for w in waves)
        sd.play(buffer / max(abs(buffer)), samplerate=fs)
        sd.wait()

    elif mode == 'random':
        random.shuffle(chord)
        waves = []
        for i, n in enumerate(chord):
            wave = generate_wave(midi_to_freq(n), duration, (duration/len(chord)), waveform)
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

    warmth, brightness, texture = get_visual_features(frame)
    chord = select_chord_by_color(warmth)
    mode = random.choice(MODES)
    waveform = select_waveform(texture)

    if brightness >= 0.75:
        duration = 1.0
    elif brightness >= 0.5:
        duration = 1.5
    elif brightness >= 0.25:
        duration = 2.0
    else:
        duration = 2.5

    print(f"Warmth: {warmth:.2f} | Brightness: {brightness:.2f} | Texture: {texture:.3f} | Mode: {mode} | Wave: {waveform} | Chord: {chord}")
    play_chord(chord, mode, duration, waveform)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
