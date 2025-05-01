// main_nocv_fixed.cpp — BBB Real-Time YUYV Capture + Audio with Extended Debug Prints
// Features: dynamic resolution, dual buffers, error checks, RT scheduling,
// graceful shutdown, debug output for brightness, warmth, texture, objCount, and audio phase

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
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <portaudio.h>
#include <algorithm>  // for std::clamp

using uchar = unsigned char;  // alias for pixel data
#include <pthread.h>

// Shared state
struct Shared {
    std::atomic<bool> running{true};
    std::atomic<int> frameCount{0};
} shared;

// Debug metrics
static std::atomic<float> avgBrightness{0.0f};
static std::atomic<float> warmth{0.0f};
static std::atomic<float> texture{0.0f};
static std::atomic<int> objCount{0};

// Audio data
struct PaData {
    std::vector<float> sine;
    std::atomic<int> phase{0};
};
static PaData paData;
static PaStream* paStream = nullptr;
static std::mutex audioMutex;

// Camera buffers
struct Buffer { void* start; size_t length; };
static int camFd = -1;
static std::vector<Buffer> camBufs;
static int camWidth = 0, camHeight = 0;

// Signal handler to stop
void handleSignal(int) {
    shared.running = false;
}

// Error helper
void perrorExit(const char* msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

// Initialize camera for YUYV capture
void initCamera(const char* dev = "/dev/video0") {
    camFd = open(dev, O_RDWR | O_NONBLOCK);
    if (camFd < 0) perrorExit("open video device");

    v4l2_capability cap{};
    if (ioctl(camFd, VIDIOC_QUERYCAP, &cap) < 0) perrorExit("VIDIOC_QUERYCAP");
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) perrorExit("Not a video capture device");

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 160;
    fmt.fmt.pix.height = 120;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(camFd, VIDIOC_S_FMT, &fmt) < 0) perrorExit("VIDIOC_S_FMT");

    camWidth = fmt.fmt.pix.width;
    camHeight = fmt.fmt.pix.height;
    std::cout << "[Camera] Resolution set: " << camWidth << "x" << camHeight << "\n";

    v4l2_requestbuffers req{};
    req.count = 2;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
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

// Cleanup camera
void closeCamera() {
    if (camFd < 0) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(camFd, VIDIOC_STREAMOFF, &type);
    for (auto& b : camBufs) munmap(b.start, b.length);
    close(camFd);
    camFd = -1;
}

// Thread: grab & debug-process frames
void videoThread() {
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
        uint32_t edge_count=0;
        int W=camWidth, H=camHeight;
        const int T = 20;
        auto clamp8 = [](int x){ return x<0?0:(x>255?255:x); };
        for (int i = 0; i < W*H*2; i += 4) {
            int Y0=data[i], U=data[i+1], Y1=data[i+2], V=data[i+3];
            sumY += Y0 + Y1;
            int ur = U - 128, vr = V - 128;
            int R0=clamp8(Y0 + ((1436*vr)>>10));
            int G0=clamp8(Y0 - (((352*ur + 731*vr)>>10)));
            int B0=clamp8(Y0 + ((1814*ur)>>10));
            int R1=clamp8(Y1 + ((1436*vr)>>10));
            int G1=clamp8(Y1 - (((352*ur + 731*vr)>>10)));
            int B1=clamp8(Y1 + ((1814*ur)>>10));
            sumR  += R0 + R1;
            sumGB += (G0 + B0) + (G1 + B1);
            if (i >= 4) {
                if (abs(Y0 - data[i-2]) > T) edge_count++;
                if (abs(Y1 - data[i-1]) > T) edge_count++;
            }
        }
        int pixels = W * H;
        float bright = sumY / float(pixels * 255);
        bright = std::clamp(bright, 0.0f, 1.0f);
        float wmean = sumR / float(pixels);
        float gbmean= sumGB / float(pixels);
        float w = (2.f*wmean) / (gbmean + 1.f);
        w = std::clamp(w, 0.f, 1.f);
        float tex = edge_count / float(2 * pixels);
        tex = std::clamp(tex, 0.f, 1.f);
        int objC = int(tex * 25.f);

        avgBrightness = bright;
        warmth       = w;
        texture      = tex;
        objCount     = objC;
        int fnum = ++shared.frameCount;
        std::cout << "[Video] Frame " << fnum
                  << " brightness=" << bright
                  << " warmth=" << w
                  << " texture=" << tex
                  << " objCount=" << objC << "\n";

        if (ioctl(camFd, VIDIOC_QBUF, &buf) < 0) perror("VIDIOC_QBUF");
    }
}

// PortAudio callback: play sine and debug phase
static int paCallback(const void*, void* output,
                      unsigned long frames,
                      const PaStreamCallbackTimeInfo*,
                      PaStreamCallbackFlags, void*)
{
    float* out = static_cast<float*>(output);
    std::lock_guard<std::mutex> lock(audioMutex);
    for (unsigned i = 0; i < frames; ++i) {
        out[2*i] = out[2*i+1] = paData.sine[paData.phase];
        paData.phase = (paData.phase + 1) % paData.sine.size();
    }
    return paContinue;
}

void initAudio() {
    PaError err = Pa_Initialize();
    if (err != paNoError) perrorExit(Pa_GetErrorText(err));

    int N = 200;
    paData.sine.resize(N);
    for (int i = 0; i < N; ++i)
        paData.sine[i] = std::sin((2*M_PI*i)/N);

    err = Pa_OpenDefaultStream(&paStream,
        0, 2, paFloat32, 44100, 64, paCallback, nullptr);
    if (err != paNoError) perrorExit(Pa_GetErrorText(err));

    err = Pa_StartStream(paStream);
    if (err != paNoError) perrorExit(Pa_GetErrorText(err));
}

int main() {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0) perror("mlockall");

    signal(SIGINT, handleSignal);
    signal(SIGTERM, handleSignal);

    initCamera();
    initAudio();

    std::thread vt(videoThread);
    struct sched_param sp;
    sp.sched_priority = 50;
    pthread_setschedparam(vt.native_handle(), SCHED_FIFO, &sp);

    // Debug loop: print last metrics continuously
    while (shared.running) {
        std::cout << "[Debug] brightness=" << avgBrightness.load()
                  << ", warmth=" << warmth.load()
                  << ", texture=" << texture.load()
                  << ", objCount=" << objCount.load()
                  << ", audio phase=" << paData.phase.load()
                  << "\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    vt.join();
    Pa_StopStream(paStream);
    Pa_CloseStream(paStream);
    Pa_Terminate();
    closeCamera();
    return 0;
}
