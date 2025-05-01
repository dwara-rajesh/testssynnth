// main_nocv_full.cpp — BBB Real-Time YUYV Capture + Audio Integration
// Features: v4l2 YUYV capture, feature extraction, chord synthesis, PortAudio playback

#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <signal.h>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <array>
#include <portaudio.h>
#include <pthread.h>

using uchar = unsigned char;

// Audio constants
static constexpr int FS = 48000;                 // 48 kHz
static constexpr int FRAMES_PER_BUFFER = 512;     // Buffer size for audio callback
static constexpr double DURATION = 4.0;
static constexpr double SUB = DURATION / 6.0;
static constexpr int SUB_FRAMES = int(SUB * FS);
static constexpr int DLY = int(SUB * FS * 0.75);

// Frequency LUT for MIDI notes
static std::array<double, 128> freqLUT;
static void initFreqLUT() {
    for (int i = 0; i < 128; ++i)
        freqLUT[i] = 440.0 * std::pow(2.0, (i - 69) / 12.0);
}

// Shared state
struct Shared {
    std::atomic<bool> running{false};
    std::deque<std::vector<float>> audioQueue;
    std::mutex seqMutex;
    std::vector<int> seq;
    std::vector<std::vector<int>> CHORDS;
    std::atomic<float> brightness{0}, warmth{0}, texture{0};
    std::atomic<int> objCount{0};
    std::atomic<int> waveform{0};
} shared;

std::mt19937 rng(std::random_device{}());

// V4L2 camera buffers
struct Buffer { void* start; size_t length; };
static int camFd = -1;
static std::vector<Buffer> camBufs;
static int camW = 0, camH = 0;

void perrorExit(const char* msg) { perror(msg); exit(EXIT_FAILURE); }

// Initialize camera for YUYV capture
void initCamera(const char* dev = "/dev/video0") {
    camFd = open(dev, O_RDWR | O_NONBLOCK);
    if (camFd < 0) perrorExit("open video device");

    v4l2_capability cap{};
    if (ioctl(camFd, VIDIOC_QUERYCAP, &cap) < 0) perrorExit("VIDIOC_QUERYCAP");
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) perrorExit("Not video capture");

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 160;
    fmt.fmt.pix.height = 120;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(camFd, VIDIOC_S_FMT, &fmt) < 0) perrorExit("VIDIOC_S_FMT");

    camW = fmt.fmt.pix.width;
    camH = fmt.fmt.pix.height;
    std::cout << "[Camera] " << camW << "x" << camH << " YUYV\n";

    v4l2_requestbuffers req{};
    req.count = 2;
    req.type = fmt.type;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(camFd, VIDIOC_REQBUFS, &req) < 0) perrorExit("VIDIOC_REQBUFS");

    camBufs.resize(req.count);
    for (unsigned i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = req.type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(camFd, VIDIOC_QUERYBUF, &buf) < 0) perrorExit("VIDIOC_QUERYBUF");
        camBufs[i].length = buf.length;
        camBufs[i].start = mmap(nullptr, buf.length,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED, camFd, buf.m.offset);
        if (camBufs[i].start == MAP_FAILED) perrorExit("mmap");
        if (ioctl(camFd, VIDIOC_QBUF, &buf) < 0) perrorExit("VIDIOC_QBUF");
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camFd, VIDIOC_STREAMON, &type) < 0) perrorExit("VIDIOC_STREAMON");
}

void closeCamera() {
    if (camFd < 0) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(camFd, VIDIOC_STREAMOFF, &type);
    for (auto& b : camBufs) munmap(b.start, b.length);
    close(camFd);
    camFd = -1;
}

// Feature extraction thread
void featureThread() {
    while (shared.running) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(camFd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) continue;
            perror("VIDIOC_DQBUF");
            break;
        }

        uchar* data = static_cast<uchar*>(camBufs[buf.index].start);
        uint64_t sumY=0, sumR=0, sumGB=0;
        uint32_t edges=0;
        int W=camW, H=camH, T=20;
        auto clamp8 = [](int x){ return x<0?0:(x>255?255:x); };

        for (int i = 0; i < W*H*2; i += 4) {
            int Y0=data[i], U=data[i+1], Y1=data[i+2], V=data[i+3];
            sumY += Y0 + Y1;
            int ur=U-128, vr=V-128;
            int R0=clamp8(Y0 + ((1436*vr)>>10));
            int G0=clamp8(Y0 - (((352*ur+731*vr)>>10)));
            int B0=clamp8(Y0 + ((1814*ur)>>10));
            sumR += R0;
            sumGB += G0 + B0;
            if (i>=4) {
                if (std::abs(Y0-data[i-2])>T) edges++;
                if (std::abs(Y1-data[i-1])>T) edges++;
            }
        }

        int pixels = W*H;
        float bright = std::clamp(sumY/float(pixels*255),0.f,1.f);
        float w = (2.f*(sumR/float(pixels))) / ((sumGB/float(pixels))+1.f);
        float tex = std::clamp(edges/float(2*pixels),0.f,1.f);
        int objCount = int(tex*25);

        shared.brightness = bright;
        // Debug print features
        std::cout << "[Features] brightness=" << bright
                  << " warmth=" << shared.warmth
                  << " texture=" << shared.texture
                  << " objCount=" << shared.objCount << std::endl;
        shared.warmth = w;
        shared.texture = tex;
        shared.objCount = objCount;

        ioctl(camFd, VIDIOC_QBUF, &buf);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// Waveform generation
static std::vector<float> generateWave(double freq,int len,double amp,int kind) {
    std::vector<float> out(len);
    for(int i=0;i<len;i++){
        double phase = 2*M_PI*freq*(i/ double(FS));
        double v=0;
        if(kind==0) v=std::sin(phase);
        else if(kind==1) { double x=2*((i/ double(FS))*freq - std::floor((i/ double(FS))*freq +0.5)); v=1-2*std::abs(x);}        
        else if(kind==2) v=(std::sin(phase)>=0?1:-1);
        out[i]=amp*v;
    }
    if(!out.empty()) out[0]=0;
    return out;
}

// Chord thread
void chordThread() {
    std::uniform_int_distribution<int> md(0,2);
    while (shared.running) {
        std::vector<int> chord;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); chord = shared.seq; }
        if (md(rng)==1) std::shuffle(chord.begin(),chord.end(),rng);
        int offs = int((shared.brightness-0.5f)*12);
        for(auto &n: chord) n += offs;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); shared.seq = chord; }
        std::this_thread::sleep_for(std::chrono::milliseconds(int(SUB*1000)));
    }
}

// Player thread
void playerThread() {
    while (shared.running) {
        std::vector<int> seq;
        { std::lock_guard<std::mutex> lk(shared.seqMutex); seq=shared.seq; }
        if(!seq.empty()){
            static int idx=0;
            int note = seq[idx%seq.size()]; idx++;
            double freq=freqLUT[note];
            // Debug print note and frequency already printed
        auto chunk=generateWave(freq,SUB_FRAMES,SUB,shared.waveform);
            std::lock_guard<std::mutex> lk(shared.seqMutex);
            if(shared.audioQueue.size()>10) shared.audioQueue.pop_front();
            shared.audioQueue.push_back(std::move(chunk));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(int(SUB*1000)));
    }
}

// PortAudio callback: mix into 16-bit stereo
static int paCallback(const void*, void* out, unsigned long frames,
                      const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*) {
    auto* buf = static_cast<int16_t*>(out);
    unsigned long totalSamples = frames*2;
    std::fill(buf, buf+totalSamples, 0);
    std::lock_guard<std::mutex> lk(shared.seqMutex);
    unsigned long i=0;
    while(i<frames && !shared.audioQueue.empty()){
        auto &chunk = shared.audioQueue.front();
        unsigned long n = std::min<unsigned long>(chunk.size(), frames-i);
        for(unsigned long j=0;j<n;++j){
            int16_t s = static_cast<int16_t>(std::clamp(chunk[j],-1.0f,1.0f)*32767);
            buf[2*(i+j)]     += s;
            buf[2*(i+j)+1]   += s;
        }
        if(n<chunk.size()) chunk.erase(chunk.begin(),chunk.begin()+n);
        else shared.audioQueue.pop_front();
        i+=n;
    }
    return paContinue;
}

int main() {
    // Setup chords
    shared.CHORDS = {
        {69,76,79,81,88,91},{62,65,69,72,74,81},{63,68,70,75,80,82},
        {65,72,75,77,84,87},{67,69,74,79,81,86},{60,64,67,71,72,79},
        {62,66,69,74,78,81},{64,68,71,76,80,83},{65,67,72,77,79,84},
        {67,71,74,79,83,86}
    };
    shared.seq = shared.CHORDS[0];
    shared.running = true;
    initFreqLUT();
    if(mlockall(MCL_CURRENT|MCL_FUTURE)<0) perror("mlockall");
    initCamera();

    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    if(err != paNoError){ std::cerr<<"Pa_Initialize error: "<<Pa_GetErrorText(err)<<"\n"; return 1; }

    PaStreamParameters outParams;
    outParams.device = Pa_GetDefaultOutputDevice();
    outParams.channelCount = 2;
    outParams.sampleFormat = paInt16;
    outParams.suggestedLatency = Pa_GetDeviceInfo(outParams.device)->defaultLowOutputLatency;
    outParams.hostApiSpecificStreamInfo = nullptr;
    err = Pa_IsFormatSupported(nullptr, &outParams, FS);
    if(err != paFormatIsSupported){ std::cerr<<"Format not supported at "<<FS<<" Hz: "<<Pa_GetErrorText(err)<<"\n"; Pa_Terminate(); return 1; }

    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream, 0, 2, paInt16, FS, FRAMES_PER_BUFFER, paCallback, nullptr);
    if(err!=paNoError){ std::cerr<<"Pa_OpenDefaultStream error: "<<Pa_GetErrorText(err)<<"\n"; Pa_Terminate(); return 1; }

    err = Pa_StartStream(stream);
    if(err!=paNoError){ std::cerr<<"Pa_StartStream error: "<<Pa_GetErrorText(err)<<"\n"; Pa_CloseStream(stream); Pa_Terminate(); return 1; }

    // Launch threads
    std::thread t1(featureThread), t2(chordThread), t3(playerThread);

    // Run
    while(shared.running) std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Cleanup
    t1.join(); t2.join(); t3.join();
    Pa_StopStream(stream); Pa_CloseStream(stream); Pa_Terminate();
    closeCamera();
    return 0;
}
