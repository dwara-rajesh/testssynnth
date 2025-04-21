// Optimized chord_synth.cpp for BeagleBone Black
// Optimizations: Lower sample rate, no GUI, simpler vision pipeline, LUTs, camera size set, fewer threads

#include <opencv2/opencv.hpp>
#include <portaudio.h>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <array>

using namespace std::chrono;

typedef float SAMPLE;
static constexpr int FS = 22050;
static constexpr double DURATION = 4.0;
static constexpr double SUB = DURATION / 6.0;
static constexpr int SUB_FRAMES = int(SUB * FS);
static constexpr int DLY = int(SUB * FS * 0.75);

// Frequency lookup table
std::array<double, 128> freqLUT;
static void initFreqLUT() {
    for (int i = 0; i < 128; ++i)
        freqLUT[i] = 440.0 * std::pow(2.0, (i - 69) / 12.0);
}

// Shared state
struct Shared {
    std::vector<std::vector<int>> CHORDS;
    std::atomic<bool> running{false};
    std::deque<std::vector<SAMPLE>> audioQueue;
    std::vector<SAMPLE> reverbTail;
    std::atomic<float> brightness{0}, warmth{0}, texture{0};
    std::atomic<int> objCount{0};
    std::atomic<float> volume{1.0f}, reverb{1.0f};
    std::atomic<int> waveform{0};
    std::mutex seqMutex;
    std::vector<int> seq;
} shared;

std::mt19937 rng(std::random_device{}());
PaStream* paStream = nullptr;
cv::VideoCapture cap(0);

static double remap(double v,double a,double b,double c,double d) { return (v - a)/(b - a)*(d - c) + c; }

static std::vector<SAMPLE> generateWave(double freq,int len,double amp,int kind) {
    std::vector<SAMPLE> out(len);
    for(int i=0;i<len;i++){
        double t = double(i) / FS;
        double phase = 2.0 * 3.14159265358979323846 * freq * t;
        double v = 0;
        if(kind == 0) v = std::sin(phase);
        else if(kind == 1) {
            double x = 2 * (t * freq - std::floor(t * freq + 0.5));
            v = (1 - 2 * std::fabs(x));
        } else if(kind == 2) {
            v = (std::sin(phase) >= 0 ? 1.0 : -1.0);
        }
        out[i] = amp * v;
    }
    out[0] = 0.0f;  // force zero-crossing at start to prevent clicks
    return out;
} else if(kind == 2) {
            v = (std::sin(2 * 3.14159265358979323846 * freq * t) >= 0 ? 1.0 : -1.0);
        }
        out[i] = amp * v;
    }
    return out;
}

static int paCallback(const void*, void* out,
    unsigned long frames, const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*) {
    SAMPLE* buf = (SAMPLE*)out;
    std::fill(buf, buf + frames, 0.0f);
    unsigned long idx = 0;
    std::lock_guard<std::mutex> lock(shared.seqMutex);
    while(idx < frames && !shared.audioQueue.empty()){
        auto& chunk = shared.audioQueue.front();
        unsigned long n = std::min<unsigned long>(chunk.size(), frames - idx);
        for(unsigned long i = 0; i < n; i++) buf[idx + i] += chunk[i];
        if(n < chunk.size()) chunk.erase(chunk.begin(), chunk.begin() + n);
        else shared.audioQueue.pop_front();
        idx += n;
    }
    return paContinue;
}

void featureThread() {
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 160);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 120);
    while(shared.running) {
        cv::Mat frame, gray;
        if(!cap.read(frame)) { std::this_thread::sleep_for(milliseconds(50)); continue; }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        float bright = cv::mean(gray)[0] / 255.0f;
        bright = std::clamp(bright, 0.0f, 1.0f);
        auto mc = cv::mean(frame);
        float r = mc[2], g = mc[1], b = mc[0];
        float denom = 2 * (b + g) + 1;
        float ratio = denom > 1e-5 ? (2 * r) / denom : 0.5f;
        float w = remap(ratio, 0.45, 0.55, 0, shared.CHORDS.size() - 0.001f);
        w = std::clamp(w, 0.0f, float(shared.CHORDS.size() - 1));
        cv::Mat edges; cv::Canny(gray, edges, 50, 150);
        float tex = float(cv::countNonZero(edges)) / (160 * 120);

        shared.brightness = bright;
        shared.warmth = w;
        shared.texture = tex;
        shared.objCount = (int)(tex * 25.0f);  // estimate based on texture
        std::this_thread::sleep_for(milliseconds(50));
    }
}

void chordWaveRevThread() {
    std::uniform_int_distribution<int> md(0, 2);
    while(shared.running) {
        // Update waveform
        float tex = shared.texture;
        shared.waveform = (tex < 0.05f ? 0 : (tex < 0.10f ? 1 : 2));

        // Update reverb
        double ratio = std::min(shared.objCount / 25.0, 1.0);
        double sens = ratio * ratio;
        double rv = remap(sens, 0, 1, 0.9, 0.1) * 0.95;
        shared.reverb = std::clamp<float>(rv, 0.0f, 1.0f);

        // Update chord
        double w = shared.warmth;
        double b = shared.brightness;
        int idx = std::min<int>(int(std::floor(w)), shared.CHORDS.size() - 1);
        const auto& chord = shared.CHORDS[idx];
        std::vector<int> modChord = chord;
        int m = md(rng);
        if(m == 0) std::reverse(modChord.begin(), modChord.end());
        else if(m == 1) std::shuffle(modChord.begin(), modChord.end(), rng);
        int offs = int(remap(b, 0, 1, -4, 3)) * 12;
        std::vector<int> seq;
        for(int n : modChord) seq.push_back(n + offs);
        seq.push_back(seq[0]);
        { std::lock_guard<std::mutex> lock(shared.seqMutex); shared.seq = seq; }

        std::this_thread::sleep_for(milliseconds(int(SUB * 1000)));
    }
}

void playerThread() {
    int idx = 0;
    while(shared.running) {
        std::vector<int> seq;
        { std::lock_guard<std::mutex> lock(shared.seqMutex); seq = shared.seq; }
        int wf = shared.waveform;
        double tex = shared.texture;
        double vol = shared.volume;
        double rev = shared.reverb;
        if(!seq.empty()) {
            int note = seq[idx % seq.size()];
            double freq = freqLUT[note];
            auto dry = generateWave(freq, SUB_FRAMES, SUB, wf);
            for(auto &s : dry) s *= vol;
            int atk = int(0.005 * FS);
            for(int i = 0; i < atk && i < (int)dry.size(); i++) dry[i] *= i / double(atk);
            int rel = int((1 - tex * tex) * dry.size());
            for(int i = 0; i < rel; i++) dry[dry.size()-1 - i] *= (rel - i) / double(rel);
            std::vector<SAMPLE> out(dry.size());
            for(size_t i = 0; i < dry.size(); i++) {
                float wet = (i < DLY ? shared.reverbTail[i] : 0.0f);
                out[i] = dry[i] * (1.0 - rev) + wet;
            }
            for(size_t i = 0; i + DLY < out.size(); i++) {
                out[i + DLY] += dry[i] * rev;
            }
            float maxv = 0;
            for(auto s : out) maxv = std::max(maxv, std::fabs(s));
            if(maxv > 0) for(auto &s : out) s /= maxv;
            int fadeLen = std::min<int>(220, out.size());  // ~10 ms at 22 kHz
            for (int i = 0; i < fadeLen; ++i) {
                float fadeIn = i / float(fadeLen);
                out[i] *= fadeIn;
                if (i < shared.reverbTail.size())
                    out[i] += shared.reverbTail[i] * (1.0f - fadeIn);
            }
            shared.reverbTail.assign(out.end() - fadeLen, out.end());
            std::lock_guard<std::mutex> lock(shared.seqMutex);
            if (shared.audioQueue.size() > 10) shared.audioQueue.pop_front();
            shared.audioQueue.push_back(std::move(out));
            idx++;
        }
        std::this_thread::sleep_for(milliseconds(int(SUB * 1000)));
    }
}

int main() {
    shared.CHORDS = {
        {69, 76, 79, 81, 88, 91}, {62, 65, 69, 72, 74, 81}, {63, 68, 70, 75, 80, 82},
        {65, 72, 75, 77, 84, 87}, {67, 69, 74, 79, 81, 86}, {60, 64, 67, 71, 72, 79},
        {62, 66, 69, 74, 78, 81}, {64, 68, 71, 76, 80, 83}, {65, 67, 72, 77, 79, 84},
        {67, 71, 74, 79, 83, 86}
    };
    shared.running = true;
    shared.reverbTail.assign(DLY, 0);
    initFreqLUT();

    Pa_Initialize();
    Pa_OpenDefaultStream(&paStream, 0, 1, paFloat32, FS, SUB_FRAMES, paCallback, nullptr);
    Pa_StartStream(paStream);

    std::thread t1(featureThread), t2(chordWaveRevThread), t3(playerThread);

    while(shared.running) {
        std::this_thread::sleep_for(milliseconds(200));
    }

    t1.join(); t2.join(); t3.join();
    Pa_StopStream(paStream); Pa_CloseStream(paStream); Pa_Terminate();
    cap.release();
    return 0;
}
