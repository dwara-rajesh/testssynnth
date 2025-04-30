// main_nocv_fixed.cpp — BBB Real-Time YUYV Capture + Audio
// Fixed: dynamic resolution, dual buffers, error checks, RT scheduling, graceful shutdown

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
#include <pthread.h>
#include <sys/mman.h>

// Shared state
typedef unsigned char uchar;
struct Shared {
    std::atomic<bool> running{true};
    std::atomic<int> seq{0};
} shared;

// Audio data
struct PaData {
    std::vector<float> sine;
    int phase = 0;
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
    if (ioctl(camFd, VIDIOC_QUERYCAP, &cap) < 0)
        perrorExit("VIDIOC_QUERYCAP");
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        perrorExit("Device not video capture");

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 160;
    fmt.fmt.pix.height = 120;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(camFd, VIDIOC_S_FMT, &fmt) < 0)
        perrorExit("VIDIOC_S_FMT");

    camWidth = fmt.fmt.pix.width;
    camHeight = fmt.fmt.pix.height;

    v4l2_requestbuffers req{};
    req.count = 2;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(camFd, VIDIOC_REQBUFS, &req) < 0)
        perrorExit("VIDIOC_REQBUFS");

    camBufs.resize(req.count);
    for (unsigned i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = req.type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(camFd, VIDIOC_QUERYBUF, &buf) < 0)
            perrorExit("VIDIOC_QUERYBUF");
        camBufs[i].length = buf.length;
        camBufs[i].start = mmap(nullptr, buf.length,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED, camFd, buf.m.offset);
        if (camBufs[i].start == MAP_FAILED)
            perrorExit("mmap");
        if (ioctl(camFd, VIDIOC_QBUF, &buf) < 0)
            perrorExit("VIDIOC_QBUF");
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camFd, VIDIOC_STREAMON, &type) < 0)
        perrorExit("VIDIOC_STREAMON");
}

// Cleanup camera
void closeCamera() {
    if (camFd < 0) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(camFd, VIDIOC_STREAMOFF, &type);
    for (auto& b : camBufs)
        munmap(b.start, b.length);
    close(camFd);
    camFd = -1;
}

// Thread: grab & stub-process frames
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
        // Stub: process YUYV at camWidth x camHeight
        // e.g., extract luma, feature detection, etc.

        if (ioctl(camFd, VIDIOC_QBUF, &buf) < 0)
            perror("VIDIOC_QBUF");

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// PortAudio callback: play sine
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
    PaError err;
    err = Pa_Initialize();
    if (err != paNoError) perrorExit(Pa_GetErrorText(err));

    // prepare sine table
    int N = 200;
    paData.sine.resize(N);
    for (int i = 0; i < N; ++i)
        paData.sine[i] = std::sin((2*M_PI*i)/N);
    paData.phase = 0;

    err = Pa_OpenDefaultStream(&paStream,
        0, 2, paFloat32, 44100, 64, paCallback, nullptr);
    if (err != paNoError) perrorExit(Pa_GetErrorText(err));

    // Consider setting RT priority using pthread_setschedparam on internal threads

    err = Pa_StartStream(paStream);
    if (err != paNoError) perrorExit(Pa_GetErrorText(err));
}

int main() {
    // Lock memory to avoid paging
    if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0)
        perror("mlockall");

    // Signal handler
    signal(SIGINT, handleSignal);
    signal(SIGTERM, handleSignal);

    initCamera();
    initAudio();

    // Launch video thread with RT priority
    std::thread vt(videoThread);
    struct sched_param sp;
    sp.sched_priority = 50;
    pthread_setschedparam(vt.native_handle(), SCHED_FIFO, &sp);

    // Wait until interrupted
    while (shared.running)
        std::this_thread::sleep_for(std::chrono::seconds(1));

    // Clean up
    vt.join();
    Pa_StopStream(paStream);
    Pa_CloseStream(paStream);
    Pa_Terminate();
    closeCamera();
    return 0;
}