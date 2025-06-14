# SynesthesiARP

![GitHub repo size](https://img.shields.io/github/repo-size/venk-meg/testssynnth)
![GitHub contributors](https://img.shields.io/github/contributors/venk-meg/testssynnth)
![GitHub forks](https://img.shields.io/github/forks/venk-meg/testssynnth)
![GitHub stars](https://img.shields.io/github/stars/venk-meg/testssynnth)
![GitHub issues](https://img.shields.io/github/issues/venk-meg/testssynnth)
![GitHub license](https://img.shields.io/github/license/venk-meg/testssynnth)

> **SynesthesiARP** is a self-contained, real-time arpeggiating synthesizer that converts live video into rich audio, designed for portability and low-latency performance on embedded hardware.

[View Demo](https://www.youtube.com/watch?v=1fKRt8jVq0s
)

---

## Table of Contents

* [About the Project](#about-the-project)
* [Hardware](#hardware)
* [Software Architecture](#software-architecture)
* [Getting Started](#getting-started)
* [Performance](#performance)
* [Future Work](#future-work)
* [Acknowledgments](#acknowledgments)

---

## About The Project

SynesthesiARP transforms **visual information** from a live camera feed into **real-time audio arpeggios** using a BeagleBone Black. The goal was to build a standalone vision-driven synth that could:

* Run efficiently with minimal CPU and memory footprint.
* Deliver sub-20 ms latency audio with rich sound quality.
* Provide a platform for accessible, interactive, multisensory experiences.

### Built With

* [OpenCV](https://opencv.org/)
* [ALSA](https://www.alsa-project.org/)
* [Buildroot](https://buildroot.org/)
* Custom Linux Kernel (v5.10.168)

---

## Hardware

* **BeagleBone Black**
* **Logitech c390e USB webcam**
* **JBL Go USB speaker**
* Custom bootable SD card (minimal kernel + rootfs)

---

## Software Architecture

| Component          | Original                       | Final                        |
| ------------------ | ------------------------------ | ---------------------------- |
| Frame resolution   | 320×240                        | 160×120                      |
| Texture extraction | Laplacian σ                    | Canny edge density           |
| Warmth metric      | RGB average                    | Channel-mean ratio remap     |
| Sequencing         | Multi-thread w/ coarse mutexes | Unified thread w/ atomic ops |
| Audio I/O          | PortAudio (jitter)             | Direct ALSA (low latency)    |

**Threads**

* **Vision Thread:** Captures 160x120 frames (\~20 FPS), extracts brightness, warmth, texture, object count.
* **Audio Thread:** Generates 6-note arpeggios mapped from features using lookup tables, attack envelope, LPF, and feedback reverb.

---

## Getting Started

### Prerequisites

* ARM cross-compiler (e.g., `arm-linux-gnueabihf-gcc`)
* Buildroot (configured for BeagleBone Black)
* Linux host (or WSL on Windows)

### Installation

1. **Kernel**

   ```bash
   git clone https://github.com/beagleboard/linux.git
   cd linux
   make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- bb.org_defconfig
   make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- menuconfig
   # Remove drivers as needed
   make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage modules dtbs
   ```

2. **Buildroot**

   ```bash
   git clone https://github.com/buildroot/buildroot.git
   cd buildroot
   make beaglebone_defconfig
   make menuconfig  # Enable OpenCV, ALSA; disable unnecessary packages
   make
   ```

3. **Application**

   * Cross-compile on host or build natively on the BeagleBone Black.

4. **Deploy**

   * Flash SD card with the Buildroot image.
   * Boot the BeagleBone Black.
   * Run the binary.

---

## Performance

| Metric            | Result                           |
| ----------------- | -------------------------------- |
| Memory usage      | < 10 MB                          |
| CPU usage         | \~90% peak (short bursts)        |
| Vision FPS        | \~20 FPS (stable with ±4 jitter) |
| Audio period rate | \~94 Hz (target: \~86 Hz)        |
| Latency           | Sub-20 ms                        |

---

## Future Work

✅ Adaptive feature thresholds (ML-based)
✅ Auto-start on boot under non-root user
✅ Power management optimization
✅ Hardware abstraction for other platforms (e.g., Raspberry Pi)
✅ Additional audio effects (e.g., convolution reverb, spatialization)
✅ Add user feedback interface (CLI, LED indicators)

---

## Acknowledgments

* [Yuri Suzuki's Colour Chaser](https://yurisuzuki.com/archive/works/colour-chaser/)
* [BeagleBoard](https://beagleboard.org/)
* [OpenCV Developers](https://opencv.org/)
* [The ALSA Project](https://www.alsa-project.org/)

---

Navigating Repository
- /Hardware has all configuration files & disk image file
- /Software has source code, binary file, and make command
- /Test has test benchmark shell script for [erformance metric analysis

Setting up your own BeagleBone Black
- Flash sdcardfinal.img into BeagleBone Black board from /Hardware folder
- Power BeagleBone Black board
- Login as root
- Enter "./full_synth" to run the application and enjoy SynesthesiARP
- OR
- Enter "./run_benchmark.sh" to log the outputs for performance analysis
