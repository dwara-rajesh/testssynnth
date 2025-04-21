// chord_synth.cpp
// C++ port of the Python streaming ARP synth with real-time webcam features
// Uses OpenCV for vision and PortAudio for audio

#include <opencv2/opencv.hpp>
#include <portaudio.h>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <random>
#include <chrono>
#include <iostream>

using namespace std::chrono;

typedef float SAMPLE;
static constexpr int FS = 44100;
static constexpr double DURATION = 4.0;
static constexpr double SUB = DURATION / 6.0;
static constexpr int SUB_FRAMES = int(SUB * FS);
static constexpr int DLY = int(SUB * FS * 0.75);

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

// Helpers
static double midiToFreq(int m) { return 440.0 * std::pow(2.0, (m - 69) / 12.0); }
static double remap(double v,double a,double b,double c,double d) { return (v-a)/(b-a)*(d-c)+c; }

// Waveform generator
static std::vector<SAMPLE> generateWave(double freq,int len,double amp,int kind) {
    std::vector<SAMPLE> out(len);
    for(int i=0;i<len;i++){
        double t = double(i)/FS;
        double v=0;
        if(kind==0) v = std::sin(2*M_PI*freq*t);
        else if(kind==1) {
            double x = 2*(t*freq - std::floor(t*freq+0.5));
            v = (1 - 2*std::fabs(x));
        } else if(kind==2) v = (std::sin(2*M_PI*freq*t)>=0?1.0:-1.0);
        out[i] = amp * v;
    }
    return out;
}

// Audio callback
static int paCallback(const void*, void* out,
    unsigned long frames, const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*) {
    SAMPLE* buf = (SAMPLE*)out;
    std::fill(buf, buf+frames, 0.0f);
    unsigned long idx=0;
    std::lock_guard<std::mutex> lock(shared.seqMutex);
    while(idx<frames && !shared.audioQueue.empty()){
        auto& chunk = shared.audioQueue.front();
        unsigned long n = std::min<unsigned long>(chunk.size(), frames-idx);
        for(unsigned long i=0;i<n;i++) buf[idx+i] += chunk[i];
        if(n<chunk.size()){
            chunk.erase(chunk.begin(), chunk.begin()+n);
        } else {
            shared.audioQueue.pop_front();
        }
        idx += n;
    }
    return paContinue;
}

// Feature thread
void featureThread(){
    while(shared.running){
        cv::Mat frame, small, gray;
        if(!cap.read(frame)){ std::this_thread::sleep_for(milliseconds(50)); continue; }
        cv::resize(frame, small, {160,120});
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        float bright = cv::mean(gray)[0]/255.0f;
        auto mc = cv::mean(small); float r=mc[2], g=mc[1], b=mc[0];
        float w = remap((2*r)/(2*(b+g)+1),0.45,0.55,0, shared.CHORDS.size()-0.001);
        cv::Mat edges; cv::Canny(gray, edges,50,150);
        float tex = float(cv::countNonZero(edges)) / (160*120);
        cv::Mat thresh;
        cv::adaptiveThreshold(gray, thresh,255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,11,2);
        cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, cv::Mat::ones(3,3,CV_8U));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        shared.brightness = bright;
        shared.warmth = w;
        shared.texture = tex;
        shared.objCount = (int)contours.size();
        std::this_thread::sleep_for(milliseconds(50));
    }
}

// Chord thread
void chordThread(){
    std::uniform_int_distribution<int> md(0,2);
    while(shared.running){
        double w = shared.warmth, b = shared.brightness;
        int idx = std::min<int>(int(std::floor(w)), shared.CHORDS.size()-1);
        auto chord = shared.CHORDS[idx];
        int m = md(rng);
        if(m==0) std::reverse(chord.begin(), chord.end());
        else if(m==1) std::shuffle(chord.begin(), chord.end(), rng);
        int offs = int(remap(b,0,1,-4,4))*12;
        std::vector<int> seq;
        for(int n:chord) seq.push_back(n+offs);
        seq.push_back(seq[0]);
        { std::lock_guard<std::mutex> lock(shared.seqMutex); shared.seq = seq; }
        std::this_thread::sleep_for(duration_cast<milliseconds>(milliseconds(int(DURATION*1000))));
    }
}

// Waveform thread
void waveThread(){
    while(shared.running){
        float tex = shared.texture;
        shared.waveform = (tex<0.06f?0:(tex<0.12f?1:2));
        std::this_thread::sleep_for(duration_cast<milliseconds>(milliseconds(int(SUB*1000))));
    }
}

// Reverb thread
void revThread(){
    while(shared.running){
        double r = std::min(shared.objCount/25.0,1.0);
        double s = r*r;
        shared.reverb = std::max(0.0, std::min(remap(s,0,1,0.9,0.1),1.0));
        std::this_thread::sleep_for(seconds(1));
    }
}

// Player thread
void playerThread(){
    int idx=0;
    while(shared.running){
        std::vector<int> seq;
        { std::lock_guard<std::mutex> lock(shared.seqMutex); seq = shared.seq; }
        int wf = shared.waveform;
        double tex = shared.texture;
        double vol = shared.volume;
        double rev = shared.reverb;
        if(!seq.empty()){
            int note = seq[idx % seq.size()];
            double freq = midiToFreq(note);
            auto dry = generateWave(freq, SUB_FRAMES, SUB, wf);
            int atk = int(0.005*FS);
            for(int i=0;i<atk && i<dry.size();i++) dry[i] *= i/(double)atk;
            int rel = int((1-tex*tex)*dry.size());
            for(int i=0;i<rel;i++) dry[dry.size()-1-i] *= (rel-i)/(double)rel;

            // new: apply continuous reverb tail
            std::vector<SAMPLE> out(dry.size());
            for(size_t i=0;i<dry.size();i++) {
                float wet = (i < DLY ? shared.reverbTail[i] : 0.0f);
                out[i] = dry[i] * (1.0 - rev) + wet;
            }
            // add fresh echoes
            for(size_t i=0;i+ DLY<out.size();i++) {
                out[i + DLY] += dry[i] * rev;
            }
            // update tail
            shared.reverbTail.assign(out.end()-DLY, out.end());

            std::lock_guard<std::mutex> lock(shared.seqMutex);
            shared.audioQueue.push_back(out);
            idx++;
        }
        std::this_thread::sleep_for(duration_cast<milliseconds>(milliseconds(int(SUB*1000))));
    }
}

int main(){
    // initialize chords
    shared.CHORDS = {
        {69,75,79,69+12,75+12,79+12}, {62,65,69,72,62+12,69+12},
        {63,69,73,63+12,69+12,73+12}, {65,72,75,65+12,72+12,75+12},
        {67,74,77,67+12,74+12,77+12}, {60,64,67,71,60+12,67+12},
        {62,66,69,62+12,66+12,69+12}, {64,68,71,75,64+12,71+12},
        {65,67,72,65+12,67+12,72+12}, {67,71,74,78,67+12,74+12}
    };
    shared.running = true;
    shared.reverbTail.assign(DLY,0);

    Pa_Initialize();
    Pa_OpenDefaultStream(&paStream, 0, 1, paFloat32, FS, SUB_FRAMES,
        paCallback, nullptr);
    Pa_StartStream(paStream);

    std::thread t1(featureThread), t2(chordThread), t3(waveThread), t4(revThread), t5(playerThread);

    cv::namedWindow("Cam");
    while(shared.running){
        cv::Mat f;
        if(!cap.read(f)) break;
        cv::imshow("Cam",f);
        if(cv::waitKey(1)==27) shared.running=false;
    }

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    Pa_StopStream(paStream); Pa_CloseStream(paStream); Pa_Terminate();
    cap.release();
    return 0;
}
