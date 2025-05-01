// main.cpp — BeagleBone Black (or any Linux) optimized synth + V4L2 capture
// No OpenCV, single‐buffer mmap, integer math, fixed resolution.

#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>
#include <iostream>

#include <portaudio.h>

using namespace std;

// audio parameters
typedef float SAMPLE;
static constexpr int FS = 22050;
static constexpr double DURATION = 4.0;
static constexpr double SUB      = DURATION / 6.0;
static constexpr int SUB_FRAMES = int(SUB * FS);
static constexpr int DLY       = int(SUB * FS * 0.75);

// frequency LUT
static array<double,128> freqLUT;
static void initFreqLUT(){
    for(int i=0;i<128;i++)
        freqLUT[i] = 440.0 * pow(2.0,(i-69)/12.0);
}

// shared state
struct Shared {
    vector<vector<int>> CHORDS;
    atomic<bool> running{false};
    deque<vector<SAMPLE>> audioQueue;
    vector<SAMPLE> reverbTail;
    atomic<float> brightness{0}, warmth{0}, texture{0};
    atomic<int> objCount{0};
    atomic<float> volume{1.0f}, reverb{1.0f};
    atomic<int> waveform{0};
    mutex seqMutex;
    vector<int> seq;
} shared;

// PortAudio stream
PaStream* paStream = nullptr;

// random
mt19937 rng(random_device{}());

// helper remap
static double remap(double v,double a,double b,double c,double d){
    return (v-a)/(b-a)*(d-c)+c;
}

// generate one waveform, zero‐cross start
static vector<SAMPLE> generateWave(double freq,int len,double amp,int kind){
    vector<SAMPLE> out(len);
    for(int i=0;i<len;i++){
        double t = double(i)/FS;
        double phase = 2.0*M_PI*freq*t;
        double v = 0;
        if(kind==0)        v = sin(phase);
        else if(kind==1){  // triangle
            double x = 2*(t*freq - floor(t*freq+0.5));
            v = 1-2*fabs(x);
        }
        else if(kind==2)   v = sin(phase)>=0 ? 1.0 : -1.0;
        out[i] = SAMPLE(amp*v);
    }
    if(!out.empty()) out[0]=0;
    return out;
}

// PortAudio callback
static int paCallback(const void*,void* output,
    unsigned long frames,const PaStreamCallbackTimeInfo*,
    PaStreamCallbackFlags,unsigned){
    auto* buf = (SAMPLE*)output;
    memset(buf,0,frames*sizeof(SAMPLE));
    unsigned long idx=0;
    lock_guard<mutex> lk(shared.seqMutex);
    while(idx<frames && !shared.audioQueue.empty()){
        auto &chunk = shared.audioQueue.front();
        unsigned long n = min<unsigned long>(chunk.size(), frames-idx);
        for(unsigned long i=0;i<n;i++) buf[idx+i]+=chunk[i];
        if(n<chunk.size()) chunk.erase(chunk.begin(),chunk.begin()+n);
        else shared.audioQueue.pop_front();
        idx+=n;
    }
    return paContinue;
}

// V4L2 capture
static int v4l2_fd=-1;
struct V4Buf{ void* start; size_t len; } vbuf;

bool initCamera(){
    v4l2_fd = open("/dev/video0", O_RDWR);
    if(v4l2_fd<0) return false;
    // set format
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 160;
    fmt.fmt.pix.height= 120;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if(ioctl(v4l2_fd, VIDIOC_S_FMT, &fmt)<0) return false;
    // request 1 buffer
    v4l2_requestbuffers req{};
    req.count=1; req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory=V4L2_MEMORY_MMAP;
    if(ioctl(v4l2_fd, VIDIOC_REQBUFS, &req)<0) return false;
    // query & mmap
    v4l2_buffer buf{};
    buf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory=V4L2_MEMORY_MMAP;
    buf.index=0;
    if(ioctl(v4l2_fd, VIDIOC_QUERYBUF, &buf)<0) return false;
    vbuf.len = buf.length;
    vbuf.start = mmap(nullptr, buf.length, PROT_READ|PROT_WRITE,
                     MAP_SHARED, v4l2_fd, buf.m.offset);
    ioctl(v4l2_fd, VIDIOC_QBUF, &buf);
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(v4l2_fd, VIDIOC_STREAMON, &type);
    return true;
}

void closeCamera(){
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(v4l2_fd, VIDIOC_STREAMOFF, &type);
    munmap(vbuf.start, vbuf.len);
    close(v4l2_fd);
}

// feature extraction thread
void featureThread(){
    const int W=160, H=120;
    const int T=20;
    v4l2_buffer bufinfo{};
    bufinfo.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufinfo.memory=V4L2_MEMORY_MMAP;

    while(shared.running){
        if(ioctl(v4l2_fd, VIDIOC_DQBUF, &bufinfo)<0){
            this_thread::sleep_for(chrono::milliseconds(50));
            continue;
        }
        uint8_t* data = (uint8_t*)vbuf.start;
        uint32_t sumY=0, edge_count=0;
        uint64_t sumR=0, sumGB=0;

        // process YUYV
        for(int i=0;i<W*H*2;i+=4){
            uint8_t Y0=data[i], U=data[i+1];
            uint8_t Y1=data[i+2],V=data[i+3];
            sumY += Y0 + Y1;
            int ur=int(U)-128, vr=int(V)-128;
            auto clamp8 = [](int x){ return x<0?0:(x>255?255:x); };
            int R0=clamp8(int(Y0)+(1436*vr>>10));
            int G0=clamp8(int(Y0)-((352*ur+731*vr)>>10));
            int B0=clamp8(int(Y0)+(1814*ur>>10));
            int R1=clamp8(int(Y1)+(1436*vr>>10));
            int G1=clamp8(int(Y1)-((352*ur+731*vr)>>10));
            int B1=clamp8(int(Y1)+(1814*ur>>10));
            sumR  += R0+R1;
            sumGB += (G0+B0)+(G1+B1);
            // horizontal edge on luma
            if(i>=4){
                if(abs(int(Y0)-int(data[i-2]))>T) edge_count++;
                if(abs(int(Y1)-int(data[i-1]))>T) edge_count++;
            }
        }

        // compute
        float bright = sumY / float(W*H*255);
        bright = clamp(bright,0.0f,1.0f);
        float wmean = float(sumR/float(W*H));
        float gbmean= float(sumGB/float(W*H));
        float warmth = (2.f*wmean)/(gbmean+1.f);
        warmth = clamp(warmth, 0.f, float(shared.CHORDS.size()-1));
        float tex = edge_count / float(2*W*H);
        shared.brightness = bright;
        shared.warmth     = warmth;
        shared.texture    = tex;
        shared.objCount   = int(tex*25.f);

        ioctl(v4l2_fd, VIDIOC_QBUF, &bufinfo);
        this_thread::sleep_for(chrono::milliseconds(50));
    }
}

// chord & reverb thread
void chordWaveRevThread(){
    uniform_int_distribution<int> md(0,2);
    while(shared.running){
        float tex = shared.texture;
        shared.waveform = tex<0.05f?0:(tex<0.10f?1:2);
        double ratio = min(shared.objCount/25.0,1.0);
        double rv = remap(ratio*ratio,0,1,0.9,0.1)*0.95;
        shared.reverb = clamp<float>(rv,0,1);
        double w = shared.warmth, b=shared.brightness;
        int idx = min<int>(floor(w), shared.CHORDS.size()-1);
        auto chord = shared.CHORDS[idx];
        int m = md(rng);
        if(m==0) reverse(chord.begin(),chord.end());
        else if(m==1) shuffle(chord.begin(),chord.end(),rng);
        int offs = int(remap(b,0,1,-4,3))*12;
        vector<int> seq;
        for(int n:chord) seq.push_back(n+offs);
        seq.push_back(seq[0]);
        { lock_guard<mutex> lk(shared.seqMutex); shared.seq=seq; }
        this_thread::sleep_for(chrono::milliseconds(int(SUB*1000)));
    }
}

// player thread
void playerThread(){
    int idx=0;
    while(shared.running){
        vector<int> seq;
        { lock_guard<mutex> lk(shared.seqMutex); seq=shared.seq; }
        int wf = shared.waveform;
        double tex=shared.texture,vol=shared.volume,rev=shared.reverb;
        if(!seq.empty()){
            int note = seq[idx%seq.size()];
            double freq = freqLUT[note];
            auto dry    = generateWave(freq,SUB_FRAMES,SUB,wf);
            // amp env
            int atk=int(0.05*FS);
            for(int i=0;i<atk && i<dry.size();i++) dry[i]*=i/double(atk);
            int rel=int((1-tex*tex)*dry.size());
            for(int i=0;i<rel;i++) dry[dry.size()-1-i]*=(rel-i)/double(rel);
            // apply vol
            for(auto &s:dry) s*=vol;
            // reverb
            vector<SAMPLE> out(dry.size());
            for(size_t i=0;i<dry.size();i++){
                float wet = i<DLY? shared.reverbTail[i]:0;
                out[i] = dry[i]*(1-rev) + wet;
            }
            for(size_t i=0;i+ DLY<out.size();i++)
                out[i+DLY] += dry[i]*rev;
            float maxv=0;
            for(auto s:out) maxv=max(maxv, fabs(s));
            if(maxv>0) for(auto &s:out) s/=maxv;
            shared.reverbTail.assign(out.end()-DLY, out.end());
            lock_guard<mutex> lk(shared.seqMutex);
            if(shared.audioQueue.size()>10) shared.audioQueue.pop_front();
            shared.audioQueue.push_back(move(out));
            idx++;
        }
        this_thread::sleep_for(chrono::milliseconds(int(SUB*1000)));
    }
}

int main(){
    // chords
    shared.CHORDS = {
        {69,76,79,81,88,91},{62,65,69,72,74,81},{63,68,70,75,80,82},
        {65,72,75,77,84,87},{67,69,74,79,81,86},{60,64,67,71,72,79},
        {62,66,69,74,78,81},{64,68,71,76,80,83},{65,67,72,77,79,84},
        {67,71,74,79,83,86}
    };
    shared.running = true;
    shared.reverbTail.assign(DLY,0);

    if(!initCamera()){
        cerr<<"Failed to open /dev/video0: "<<strerror(errno)<<"\n";
        return 1;
    }
    initFreqLUT();

    Pa_Initialize();
    Pa_OpenDefaultStream(&paStream,0,1,paFloat32,FS,SUB_FRAMES,paCallback,nullptr);
    Pa_StartStream(paStream);

    thread t1(featureThread),
           t2(chordWaveRevThread),
           t3(playerThread);

    // run until you kill the program
    while(shared.running) this_thread::sleep_for(chrono::seconds(1));

    shared.running=false;
    t1.join(); t2.join(); t3.join();
    Pa_StopStream(paStream); Pa_CloseStream(paStream); Pa_Terminate();
    closeCamera();
    return 0;
}
