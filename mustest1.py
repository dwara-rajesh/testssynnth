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
def_rev_arp = 'rev_arp'
def_forward_arp = 'forward_arp'
MODES = [def_rev_arp, 'random', def_forward_arp, 'random']

# Waveforms selection based on texture
WAVEFORMS = ['sine', 'triangle', 'square']


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def generate_wave(freq, duration, amp, kind):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if kind == 'sine':
        wave = amp * np.sin(2 * np.pi * freq * t)
    elif kind == 'square':
        wave = amp * np.sign(np.sin(2 * np.pi * freq * t))
    elif kind == 'triangle':
        wave = amp * (2 * np.abs(2 * ((t * freq) % 1) - 1) - 1)
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

    # Warmth includes green
    warmth_ratio = (2 * r) / (2 * (b + g) + 1)
    norm_warmth = remap(warmth_ratio, 0.45, 0.55, 0.4, len(CHORDS) - 0.001)

    brightness = np.mean(gray) / 255.0

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    return norm_warmth, brightness, edge_density


def select_chord_by_color(warmth):
    idx = min(int(warmth), len(CHORDS) - 1)
    return CHORDS[idx]


def select_waveform(texture):
    if texture < 0.04:
        return 'sine'
    elif texture < 0.06:
        return 'triangle'
    else:
        return 'square'


def generate_chord_buffer(chord, mode, duration, waveform):
    # Build note sequence
    if mode == def_rev_arp:
        seq = list(reversed(chord))
    elif mode == def_forward_arp:
        seq = chord
    else:
        seq = chord.copy()
        random.shuffle(seq)

    total_len = int(fs * duration)
    buf = np.zeros(total_len, dtype=np.float32)
    for i, note in enumerate(seq):
        freq = midi_to_freq(note)
        amp = duration / len(chord)
        wave = generate_wave(freq, duration, amp, waveform)
        start = int(i * duration/len(chord) * fs)
        end = start + len(wave)
        buf[start:end] += wave[:max(0, total_len - start)]
    return buf


def crossfade_buffers(buf1, buf2, fade_len):
    """Crossfade buf1->buf2 over fade_len samples. Returns combined buffer."""
    L = max(len(buf1), len(buf2))
    out = np.zeros(L, dtype=np.float32)

    # fade-out buf1
    out[:len(buf1)] += buf1 * np.concatenate([np.ones(len(buf1) - fade_len), np.linspace(1, 0, fade_len)])
    # fade-in buf2
    out[:len(buf2)] += buf2 * np.concatenate([np.linspace(0, 1, fade_len), np.ones(len(buf2) - fade_len)])
    return out


cap = cv2.VideoCapture(0)
print("[INFO] Playing base chords from webcam input. Press ESC to quit.")

prev_buf = None
prev_duration = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    warmth, brightness, texture = get_visual_features(frame)
    chord = select_chord_by_color(warmth)
    mode = random.choice(MODES)
    waveform = select_waveform(texture)

    # brightness → duration mapping
    if brightness >= 0.75:
        duration = 2.0
    elif brightness >= 0.45:
        duration = 2.5
    else:
        duration = 3.0

    # generate buffer
    buf = generate_chord_buffer(chord, mode, duration, waveform)
    # apply fade to buf itself
    fade_len = int(0.1 * duration * fs)
    buf[:fade_len] *= np.linspace(0, 1, fade_len)
    buf[-fade_len:] *= np.linspace(1, 0, fade_len)

    if prev_buf is None:
        out_buf = buf
    else:
        # crossfade prev_buf and current buf
        out_buf = crossfade_buffers(prev_buf, buf, fade_len)

    # play combined
    sd.play(out_buf / np.max(np.abs(out_buf)), samplerate=fs, blocking=False)

    # schedule next
    sleep_time = duration - 0.2 * duration
    time.sleep(max(sleep_time, 0))
    prev_buf = buf

    print(f"Warmth: {warmth:.2f} | Bright: {brightness:.2f} | Text: {texture:.3f} | Mode: {mode} | Wave: {waveform} | Chord: {chord}")
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
