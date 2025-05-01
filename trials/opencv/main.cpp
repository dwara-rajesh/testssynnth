// main.cpp — OpenCV + PortAudio Real-Time Synth on BBB
#include <opencv2/opencv.hpp>
#include <portaudio.h>

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
static constexpr int FS                 = 48000;   // hardware native
static constexpr int FRAMES_PER_BUFFER  = 512;     // ~10.7 ms latency
static constexpr double DURATION       = 4.0;     // chord length
static constexpr double SUB            = DURATION/6.0;
static constexpr int SUB_FRAMES        = int(SUB * FS);

// MIDI → freq LUT
static double freqLUT[128];
void initFreqLUT(){
    for(int i=0;i<128;i++)
        freqLUT[i] = 440.0 * std::pow(2.0,(i-69)/12.0);
}

// Shared state
struct Shared {
    std::atomic<bool>        running{false};
    std::deque<std::vector<float>> audioQueue;
    std::mutex               seqMutex;
    std::vector<int>         seq;
    std::vector<std::vector<int>> CHORDS;
    std::atomic<int>         mode{1};       // 0=rev_arp,1=random,2=forward_arp
    std::atomic<float>       brightness{0}, warmth{0}, texture{0};
    std::atomic<int>         objCount{0};
    std::atomic<int>         waveform{0};
} shared;

// Generate basic waveform chunk
std::vector<float> generateWave(double freq,int len,double amp,int kind){
    std::vector<float> out(len);
    for(int i=0;i<len;i++){
        double t = i/(double)FS;
        double phase = 2*M_PI*freq*t;
        double v=0;
        if(kind==0) v = std::sin(phase);
        else if(kind==1){
            double frac = std::fmod(t*freq,1.0);
            double x = 2*(frac - 0.5);
            v = 1 - 2*std::abs(x);
        } else if(kind==2){
            v = (std::sin(phase)>=0?1:-1);
        }
        out[i] = float(amp * v);
    }
    // fade in/out
    int fade = std::min(len/10,50);
    for(int i=0;i<fade;i++){
        float g = float(i)/fade;
        out[i]    *= g;
        out[len-1-i] *= g;
    }
    return out;
}

// PortAudio callback: mix queued chunks into 16-bit PCM stereo
static int paCallback(
    const void* /*in*/, void* out,
    unsigned long frames,
    const PaStreamCallbackTimeInfo*,
    PaStreamCallbackFlags, void*
){
    int16_t* buf = (int16_t*)out;
    unsigned long total = frames*2;
    std::fill(buf, buf+total, 0);
    std::lock_guard<std::mutex> lk(shared.seqMutex);
    unsigned long idx=0;
    while(idx<frames && !shared.audioQueue.empty()){
        auto &chunk = shared.audioQueue.front();
        unsigned long n = std::min<unsigned long>(chunk.size(),frames-idx);
        for(unsigned long j=0;j<n;j++){
            int16_t s = int16_t(std::clamp(chunk[j],-1.0f,1.0f)*32767);
            buf[2*(idx+j)]   += s;
            buf[2*(idx+j)+1] += s;
        }
        if(n<chunk.size())
            chunk.erase(chunk.begin(),chunk.begin()+n);
        else
            shared.audioQueue.pop_front();
        idx += n;
    }
    return paContinue;
}

// Feature extractor: runs ~20 fps
void featureThread(){
    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 160);
    cap.set(CAP_PROP_FRAME_HEIGHT,120);
    if(!cap.isOpened()){ std::cerr<<"Can't open camera\n"; shared.running=false; return; }
    Mat frame;
    while(shared.running){
        auto t0 = high_resolution_clock::now();
        cap >> frame;
        if(frame.empty()) continue;
        // YUV simulation: convert to Y
        Mat yuv;
        cvtColor(frame,yuv,COLOR_BGR2YUV);
        auto ptr = yuv.ptr<uchar>();
        int W=yuv.cols, H=yuv.rows;
        uint64_t sumY=0, sumR=0,sumGB=0,edges=0;
        int T=20;
        for(int i=0;i<W*H;i++){
            int Y = ptr[i];
            sumY += Y;
            // approximate warmth/texture omitted for brevity
        }
        float bright = sumY/float(W*H)/255.f;
        // stub warmth/texture
        float w=bright, tex=0; int obj=int(tex*25);
        shared.brightness=bright; shared.warmth=w;
        shared.texture=tex; shared.objCount=obj;
        std::cout<<"[Feat] b="<<bright<<" w="<<w<<" t="<<tex<<" o="<<obj<<"\n";
        // cycle ~50 ms
        std::this_thread::sleep_until(t0 + milliseconds(50));
    }
}

// Chord manager
void chordThread(){
    std::mt19937 rng(std::random_device{}());
    while(shared.running){
        std::vector<int> chord;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); chord=shared.seq; }
        int m = shared.mode.load();
        switch(m){
            case 0: reverse(chord.begin(),chord.end());
                    std::cout<<"[Chord] rev_arp\n"; break;
            case 1: shuffle(chord.begin(),chord.end(),rng);
                    std::cout<<"[Chord] random\n"; break;
            case 2: if(!chord.empty()){
                        int f=chord.front(); chord.erase(chord.begin());
                        chord.push_back(f);
                    }
                    std::cout<<"[Chord] forward_arp\n"; break;
        }
        int offs=int((shared.brightness-0.5f)*12);
        for(auto &n:chord) n+=offs;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); shared.seq=chord; }
        std::this_thread::sleep_for(milliseconds(int(SUB*1000)));
    }
}

// Player: generates each SUB cycle
void playerThread(){
    while(shared.running){
        std::vector<int> seq;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); seq=shared.seq; }
        if(!seq.empty()){
            static size_t idx=0;
            int note = seq[idx%seq.size()]; idx++;
            double freq = freqLUT[note];
            std::cout<<"[Play] note="<<note<<" freq="<<freq<<"\n";
            auto chunk = generateWave(freq,SUB_FRAMES,SUB,shared.waveform);
            std::lock_guard<std::mutex> lk(shared.seqMutex);
            if(shared.audioQueue.size()>10) shared.audioQueue.pop_front();
            shared.audioQueue.push_back(std::move(chunk));
        }
        std::this_thread::sleep_for(milliseconds(int(SUB*1000)));
    }
}

int main(){
    // Setup chords
    shared.CHORDS = {
        {60,64,67}, {62,65,69}, {64,67,71}, {65,69,72}
    };
    shared.seq = shared.CHORDS[0];
    shared.running = true;

    initFreqLUT();

    // Init PortAudio
    PaError err = Pa_Initialize();
    if(err!=paNoError){ std::cerr<<"Pa_Init: "<<Pa_GetErrorText(err)<<"\n"; return 1; }

    // Format support check
    PaStreamParameters outParams{};
    outParams.device             = Pa_GetDefaultOutputDevice();
    outParams.channelCount      = 2;
    outParams.sampleFormat      = paInt16;
    outParams.suggestedLatency  = Pa_GetDeviceInfo(outParams.device)->defaultLowOutputLatency;
    err = Pa_IsFormatSupported(nullptr,&outParams,FS);
    if(err!=paFormatIsSupported){
        std::cerr<<"Format unsupported: "<<Pa_GetErrorText(err)<<"\n";
        Pa_Terminate(); return 1;
    }

    PaStream* stream;
    err = Pa_OpenDefaultStream(
        &stream,
        0,2,           // 0 in, 2 out
        paInt16,
        FS,FRAMES_PER_BUFFER,
        paCallback,nullptr
    );
    if(err!=paNoError){ std::cerr<<"Pa_Open: "<<Pa_GetErrorText(err)<<"\n"; Pa_Terminate(); return 1; }

    err = Pa_StartStream(stream);
    if(err!=paNoError){ std::cerr<<"Pa_Start: "<<Pa_GetErrorText(err)<<"\n"; Pa_CloseStream(stream); Pa_Terminate(); return 1; }

    // Launch threads
    std::thread t1(featureThread),
                t2(chordThread),
                t3(playerThread);

    // Run until Ctrl-C
    while(shared.running){
        std::this_thread::sleep_for(milliseconds(200));
    }

    // Cleanup
    t1.join(); t2.join(); t3.join();
    Pa_StopStream(stream); Pa_CloseStream(stream); Pa_Terminate();
    return 0;
}
