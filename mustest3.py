# chord_synth.py
import cv2
import numpy as np
import sounddevice as sd
import threading
import random
import time
from collections import deque

fs = 44100

# 10 base chords: mix of suspended and seventh voicings
CHORDS = [
    # Dark: only 'sus' with tritone (b5) or minor7
    [69, 75, 79, 69+12, 75+12, 79+12],  # A7b5 (A, D#, G)
    [62, 65, 69, 72, 62+12, 69+12],      # D minor7 (Dm7)
    [63, 69, 73, 63+12, 69+12, 73+12],   # Eb7b5 (Eb, B, Db)
    [65, 72, 75, 65+12, 72+12, 75+12],    # F minor7 (Fm7)
    [67, 74, 77, 67+12, 74+12, 77+12],    # G7b5 (G, Db, F)
    # Bright: major7 and sus2
    [60, 64, 67, 71, 60+12, 67+12],      # C major7
    [62, 66, 69, 62+12, 66+12, 69+12],    # D sus4
    [64, 68, 71, 75, 64+12, 71+12],      # E major7
    [65, 67, 72, 65+12, 67+12, 72+12],    # F sus2
    [67, 71, 74, 78, 67+12, 74+12]       # G major7
]  # A sus2
MODES = ['rev_arp', 'random', 'forward_arp']

duration = 4.0  # full bar (3/4 time)
sub = duration / 6.0  # subdivision for 6-note arp
dly = int(sub * fs * 0.75)  # reverb delay

# Shared state
global_seq = []
global_waveform = 'sine'
global_texture = 0.0
global_warmth = 0.0
global_brightness = 0.0
global_obj_count = 0
global_volume = 1.0   # amplitude
running = True
seq_lock = threading.Lock()
wav_lock = threading.Lock()
feat_lock = threading.Lock()
# continuous reverb tail and audio queue
global_reverb = 1.0   # 0.0 dry – 1.0 wet
global_reverb_tail = np.zeros(dly, dtype=np.float32)
audio_queue = deque()

# Utility lambdas
midi_to_freq = lambda m: 440.0 * 2 ** ((m - 69) / 12)
remap = lambda v,a,b,c,d: (v-a)/(b-a)*(d-c)+c

# Wave generator
def generate_wave(freq, length, amp, kind):
    t = np.linspace(0, length, int(fs*length), endpoint=False)
    if kind == 'sine': return amp * np.sin(2*np.pi*freq*t)
    if kind == 'triangle': return amp * (2*np.abs(2*((t*freq)%1)-1)-1)
    if kind == 'square': return amp * np.sign(np.sin(2*np.pi*freq*t))
    return np.zeros_like(t)

# Audio callback for continuous streaming
def audio_callback(outdata, frames, time_info, status):
    buf = np.zeros(frames, dtype=np.float32)
    idx = 0
    while idx < frames and audio_queue:
        chunk = audio_queue.popleft()
        n = min(len(chunk), frames - idx)
        buf[idx:idx+n] += chunk[:n]
        if n < len(chunk):
            audio_queue.appendleft(chunk[n:])
        idx += n
    outdata[:] = buf.reshape(-1,1)

# Start audio stream
stream = sd.OutputStream(channels=1, samplerate=fs, callback=audio_callback, blocksize= int(sub*fs))
stream.start()

# Video capture
def setup_camera():
    return cv2.VideoCapture(0)
cap = setup_camera()

# Feature collector: runs every frame
def feature_updater():
    global global_texture, global_warmth, global_brightness, global_obj_count
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        small = cv2.resize(frame, (160,120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        r,g,b = [np.mean(small[:,:,i]) for i in (2,1,0)]
        warmth = remap((2*r)/(2*(b+g)+1), 0.45,0.55,0,len(CHORDS)-0.001)
        edges = cv2.Canny(gray, 50,150)
        texture = np.count_nonzero(edges)/edges.size
        # more sensitive object segmentation:
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        kernel = np.ones((3,3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(
            clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        obj_count = len(contours)

        with feat_lock:
            global_brightness = brightness
            global_warmth = warmth
            global_texture = texture
            global_obj_count = obj_count
        time.sleep(0.05)

# Chord updater: runs every bar
def chord_updater():
    global global_seq
    while running:
        with feat_lock:
            warmth = global_warmth
            brightness = global_brightness
        chord = CHORDS[min(int(warmth), len(CHORDS)-1)]
        mode = random.choice(MODES)
        seq = chord.copy()
        if mode=='rev_arp': seq.reverse()
        elif mode=='random': random.shuffle(seq)
        offs = int(remap(brightness,0,1,-2,2))*12
        seq = [n+offs for n in seq] + [seq[0]+offs]
        with seq_lock:
            global_seq = seq
        time.sleep(duration)

# Waveform updater: runs every subdivision
def waveform_updater():
    global global_waveform
    while running:
        with feat_lock:
            texture = global_texture
        wf = 'sine' if texture<0.06 else ('triangle' if texture<0.12 else 'square')
        with wav_lock:
            global_waveform = wf
        time.sleep(sub)

# Reverb updater: maps object count
def reverb_updater():
    global global_reverb
    while running:
        with feat_lock:
            obj_count = global_obj_count
        ratio = min(obj_count/25,1.0)
        sens = ratio**2
        rv = remap(sens,0,1,0.9,0.1)
        global_reverb = max(0.0,min(rv,1.0))
        time.sleep(1)

# Player: schedules note buffers into audio_queue
def player():
    global global_reverb_tail
    global global_volume
    idx=0
    while running:
        with seq_lock:
            seq=list(global_seq)
        with wav_lock:
            wf=global_waveform
        with feat_lock:
            tex=global_texture
            vol=global_volume
            rev=global_reverb
        if seq:
            note=seq[idx%len(seq)]; freq=midi_to_freq(note)
            dry = generate_wave(freq, sub, sub, wf) * vol
            # apply attack envelope to avoid clicks
            attack_len = int(0.005 * fs)  # 5 ms fade-in
            if attack_len < len(dry):
                atk_env = np.linspace(0, 1, attack_len)
                dry[:attack_len] *= atk_env
            rel=int((1-tex**2)*len(dry))
            if rel>0:
                e=np.ones_like(dry); e[-rel:]=np.linspace(1,0,rel); dry*=e
            new_echo=dry[:-dly]*rev
            out=dry*(1-rev)
            out[:dly]+=global_reverb_tail
            out[dly:]+=new_echo
            global_reverb_tail=out[-dly:].copy()
            audio_queue.append(out/np.max(np.abs(out)))
            idx+=1
        time.sleep(sub)

# Start threads
t_feat=threading.Thread(target=feature_updater,daemon=True)
t_chord=threading.Thread(target=chord_updater,daemon=True)
t_wave=threading.Thread(target=waveform_updater,daemon=True)
t_rev=threading.Thread(target=reverb_updater,daemon=True)
t_player=threading.Thread(target=player,daemon=True)
t_feat.start();t_chord.start();t_wave.start();t_rev.start();t_player.start()

print("[INFO] Streaming ARP synth running. Esc to quit.")
while True:
    ret,frame=cap.read()
    if not ret: continue
    cv2.imshow('Cam',frame)
    if cv2.waitKey(1)&0xFF==27:
        running=False; break
cap.release(); cv2.destroyAllWindows(); stream.stop(); stream.close()
