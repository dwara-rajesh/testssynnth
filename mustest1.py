import cv2
import numpy as np
import sounddevice as sd

fs = 44100

def generate_tone(freq, duration, amp):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    wave = amp * np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)

cap = cv2.VideoCapture(0)
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    small = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray) / 255.0
    warmth = np.mean(small[:, :, 2]) / (np.mean(small[:, :, 0]) + 1)
    warmth = min(warmth / 3.0, 1.0)

    pitch = 220 + 150 * brightness
    amp = 0.2 + 0.6 * warmth
    duration = 0.9

    tone = generate_tone(pitch, duration, amp)
    sd.play(tone, samplerate=fs)
    sd.wait()

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
