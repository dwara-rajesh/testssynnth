# multi_voice_pad.py
import cv2
import numpy as np
import sounddevice as sd
import time
import threading

fs = 44100
active_voices = []
voice_lock = threading.Lock()
log_file = open("param_log.txt", "w")

# Generate smooth pad tone (with envelope)
def generate_voice(freq, amp, duration, waveform='sine'):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    if waveform == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif waveform == 'triangle':
        wave = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif waveform == 'noise':
        wave = np.random.uniform(-1, 1, t.shape)
    else:
        wave = np.zeros_like(t)

    # Simple attack-release envelope
    env = np.ones_like(t)
    attack_len = int(0.1 * fs)
    release_len = int(0.2 * fs)
    env[:attack_len] = np.linspace(0, 1, attack_len)
    env[-release_len:] = np.linspace(1, 0, release_len)

    return (amp * wave * env).astype(np.float32)

def audio_thread():
    while True:
        with voice_lock:
            if active_voices:
                # Find the max length among all voices
                max_len = max(len(v) for v in active_voices)
                padded = [np.pad(v, (0, max_len - len(v))) for v in active_voices]
                buffer = np.sum(padded, axis=0)
                active_voices.clear()
                sd.play(buffer, fs)
                sd.wait()
        time.sleep(0.05)

threading.Thread(target=audio_thread, daemon=True).start()

cap = cv2.VideoCapture(0)
print("[INFO] Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_small = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Feature extraction
    brightness = np.mean(gray) / 255.0
    warmth = np.mean(frame_small[:,:,2]) / (np.mean(frame_small[:,:,0]) + 1)
    warmth = min(warmth / 3.0, 1.0)
    edges = cv2.Canny(gray, 50, 150)
    openness = np.count_nonzero(edges) / edges.size
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = min(np.std(lap) / 100.0, 1.0)

    # Map to synth params
    base_pitch = 220 + 100 * texture
    velocity = 0.3 + 0.5 * warmth
    duration = 0.5 + 1.0 * (1 - brightness)
    waveform = 'sine' if warmth > 0.6 else ('noise' if texture > 0.6 else 'triangle')

    # Polyphony: play a triad
    intervals = [1.0, 5/4, 3/2]  # root, major third, fifth
    with voice_lock:
        for i in intervals:
            v = generate_voice(base_pitch * i, velocity, duration, waveform)
            active_voices.append(v)

    # Logging
    log_file.write(f"time={time.time():.2f}, pitch={base_pitch:.2f}, vel={velocity:.2f}, dur={duration:.2f}, wave={waveform}\n")
    log_file.flush()

    # Show webcam
    cv2.imshow("Live Visuals", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
