// main.cpp — OpenCV + PortAudio Real-Time Synth with Reverb
#include <opencv2/opencv.hpp>
#include <portaudio.h>
#include <algorithm>  // for std::clamp
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace cv;
using namespace std::chrono;

// Audio settings
static constexpr int FS                = 48000;
static constexpr int FRAMES_PER_BUFFER = 512;
static constexpr double DURATION      = 4.0;
static constexpr double SUB           = DURATION/6.0;
static constexpr int SUB_FRAMES       = int(SUB * FS);

// Reverb settings
static constexpr double REVERB_DELAY_SEC = 0.3; // 300ms delay
static const int REVERB_DELAY_SAMPLES = int(FS * REVERB_DELAY_SEC);

// MIDI → freq LUT
static double freqLUT[128];
void initFreqLUT(){
    for(int i=0;i<128;i++)
        freqLUT[i] = 440.0 * std::pow(2.0,(i-69)/12.0);
}

// Shared state
struct Shared {
    std::atomic<bool> running{false};
    std::deque<std::vector<float>> audioQueue;
    std::mutex seqMutex;
    std::vector<int> seq;
    std::vector<std::vector<int>> CHORDS;
    std::atomic<int> mode{1};    // 0=rev_arp,1=random,2=forward_arp
    std::atomic<float> brightness{0}, warmth{0}, texture{0};
    std::atomic<int> objCount{0};
    std::atomic<int> waveform{0};
    std::vector<float> reverbBuf; // circular buffer
    std::atomic<size_t> reverbPos{0};
    std::atomic<float> reverbMix{0.5f};
} shared;

std::mt19937 rng(std::random_device{}());

// Generate waveform chunk
std::vector<float> generateWave(double freq,int len,double amp,int kind){
    std::vector<float> out(len);
    for(int i=0;i<len;i++){
        double t = i/(double)FS;
        double phase = 2*M_PI*freq*t;
        double v=0;
        if(kind==0) v = sin(phase);
        else if(kind==1){ double frac=fmod(t*freq,1.0); double x=2*(frac-0.5); v=1-2*abs(x); }
        else if(kind==2) v = (sin(phase)>=0?1:-1);
        out[i] = float(amp * v);
    }
    // fade in/out
    int fade = min(len/10,50);
    for(int i=0;i<fade;i++){
        float g = float(i)/fade;
        out[i]    *= g;
        out[len-1-i] *= g;
    }
    return out;
}

// PortAudio callback: 16-bit stereo with reverb
static int paCallback(const void*, void* out, unsigned long frames,
                      const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*){
    auto* buf = static_cast<int16_t*>(out);
    unsigned long total = frames*2;
    std::fill(buf, buf+total, 0);
    std::lock_guard<std::mutex> lk(shared.seqMutex);
    unsigned long idx=0;
    for(; idx<frames && !shared.audioQueue.empty();){
        auto &chunk = shared.audioQueue.front();
        unsigned long n = std::min<unsigned long>(chunk.size(), frames-idx);
        for(unsigned long j=0;j<n;j++){
            float sample = chunk[j];
            // reverb feedback
            size_t pos = shared.reverbPos;
            float delayed = shared.reverbBuf[pos];
            float outS = sample + delayed*shared.reverbMix;
            shared.reverbBuf[pos] = outS;
            shared.reverbPos = (pos+1) % shared.reverbBuf.size();
            int16_t s = int16_t(std::clamp(outS,-1.0f,1.0f)*32767);
            buf[2*(idx+j)]   += s;
            buf[2*(idx+j)+1] += s;
        }
        if(n<chunk.size()) chunk.erase(chunk.begin(), chunk.begin()+n);
        else shared.audioQueue.pop_front();
        idx += n;
    }
    return paContinue;
}

// Feature extraction using OpenCV
void featureThread(){
    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH,160);
    cap.set(CAP_PROP_FRAME_HEIGHT,120);
    if(!cap.isOpened()){ std::cerr<<"Camera open failed"<<std::endl; shared.running=false; return; }
    Mat frame, gray;
    while(shared.running){
        auto t0 = high_resolution_clock::now();
        cap >> frame;
        if(frame.empty()) continue;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // brightness
        shared.brightness = mean(gray)[0]/255.f;
        // warmth = R channel mean
        shared.warmth = mean(Mat(frame).reshape(1).col(2))[0]/255.f;
        // texture = Laplacian variance normalized
        Mat lap;
        Laplacian(gray, lap, CV_64F);
        Scalar mu, sigma;
        meanStdDev(lap, mu, sigma);
        shared.texture = std::clamp(float(sigma[0]*sigma[0]/1000.0), 0.0f, 1.0f); // scale
        shared.texture = clamp(shared.texture,0.f,1.f);
        // object count via contours
        Mat bw;
        threshold(gray, bw,128,255,THRESH_BINARY);
        std::vector<std::vector<Point>> ctrs;
        findContours(bw, ctrs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        shared.objCount = (int)ctrs.size();
        std::cout<<"[Feat] b="<<shared.brightness
                 <<" w="<<shared.warmth
                 <<" tx="<<shared.texture
                 <<" o="<<shared.objCount<<std::endl;
        std::this_thread::sleep_until(t0 + milliseconds(50));
    }
}

// Chord thread
void chordThread(){
    while(shared.running){
        std::vector<int> chord;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); chord=shared.seq; }
        switch(shared.mode.load()){
            case 0: reverse(chord.begin(), chord.end()); break; // rev_arp
            case 1: shuffle(chord.begin(), chord.end(), rng); break; // random
            case 2: if(!chord.empty()){ int f=chord.front(); chord.erase(chord.begin()); chord.push_back(f);} break; // forward_arp
        }
        int offs = int((shared.brightness-0.5f)*12);
        for(auto &n:chord) n+=offs;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); shared.seq=chord; }
        std::this_thread::sleep_for(milliseconds(int(SUB*1000)));
    }
}

// Player thread
void playerThread(){
    while(shared.running){
        std::vector<int> seq;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); seq=shared.seq; }
        if(!seq.empty()){
            static size_t idx=0;
            int note = seq[idx%seq.size()]; idx++;
            double freq = freqLUT[note];
            std::cout<<"[Play] note="<<note<<" freq="<<freq<<std::endl;
            auto chunk = generateWave(freq,SUB_FRAMES,SUB,shared.waveform);
            std::lock_guard<std::mutex> lk(shared.seqMutex);
            if(shared.audioQueue.size()>10) shared.audioQueue.pop_front();
            shared.audioQueue.push_back(std::move(chunk));
        }
        std::this_thread::sleep_for(milliseconds(int(SUB*1000)));
    }
}

int main(){
    // init
    shared.CHORDS = {{60,64,67},{62,65,69},{64,67,71},{65,69,72}};
    shared.seq = shared.CHORDS[0];
    shared.running=true;
    shared.reverbBuf.assign(REVERB_DELAY_SAMPLES,0.0f);

    initFreqLUT();

    PaError err = Pa_Initialize();
    if(err!=paNoError){ std::cerr<<"Pa_Init:"<<Pa_GetErrorText(err)<<"\n";return 1; }

    PaStreamParameters outP{};
    outP.device            = Pa_GetDefaultOutputDevice();
    outP.channelCount      = 2;
    outP.sampleFormat      = paInt16;
    outP.suggestedLatency  = Pa_GetDeviceInfo(outP.device)->defaultLowOutputLatency;
    err = Pa_IsFormatSupported(nullptr,&outP,FS);
    if(err!=paFormatIsSupported){ std::cerr<<"Format not support"<<Pa_GetErrorText(err)<<"\n";Pa_Terminate();return 1; }

    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream, 0,2, paInt16, FS, FRAMES_PER_BUFFER, paCallback, nullptr);
    if(err!=paNoError){ std::cerr<<"Pa_Open:"<<Pa_GetErrorText(err)<<"\n"; Pa_Terminate(); return 1; }
    err = Pa_StartStream(stream);
    if(err!=paNoError){ std::cerr<<"Pa_Start:"<<Pa_GetErrorText(err)<<"\n"; Pa_CloseStream(stream); Pa_Terminate(); return 1; }

    // threads
    std::thread t1(featureThread), t2(chordThread), t3(playerThread);
    while(shared.running) std::this_thread::sleep_for(milliseconds(200));
    t1.join(); t2.join(); t3.join();

    Pa_StopStream(stream); Pa_CloseStream(stream); Pa_Terminate();
    return 0;
}