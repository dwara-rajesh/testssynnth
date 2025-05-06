/*
 * ALSA MMAP-based Real-Time Vision-Driven Chord Synth with Frequency LUT
 * Implements exact Python music logic, replaces PortAudio with ALSA.
 * Tunable parameters marked with // <<< TUNABLE >>>
 */

 #include <opencv2/opencv.hpp>
 #include <alsa/asoundlib.h>
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
 
 // <<< TUNABLE >>> Sample rate and buffer config
 static constexpr int FS                  = 48000;
 static constexpr unsigned int CHANNELS   = 1;
 static constexpr snd_pcm_format_t FORMAT = SND_PCM_FORMAT_S16_LE;
 static constexpr snd_pcm_access_t ACCESS = SND_PCM_ACCESS_MMAP_INTERLEAVED;
 static constexpr unsigned int PERIOD_FRAMES = 480;  // frames per period <<< TUNABLE >>>
 static constexpr unsigned int BUFFER_PERIODS = 4;   // total buffer = PERIOD_FRAMES * BUFFER_PERIODS <<< TUNABLE >>>
 
 // <<< TUNABLE >>> Music logic parameters
 static constexpr double DURATION = 4.0;              // full bar in seconds
 static constexpr double SUB      = DURATION / 6.0;   // subdivision
 static constexpr int SUB_FRAMES = static_cast<int>(SUB * FS);
 static constexpr int DLY_FRAMES = static_cast<int>(SUB * FS * 0.75); // reverb delay in frames
 
 // Chords exactly as Python version
 static const std::vector<std::vector<int>> CHORDS = {
     {69,75,79,81,87,91}, {62,65,69,72,74,81}, {63,69,73,75,81,85},
     {65,72,75,77,84,87}, {67,74,77,79,86,89}, {60,64,67,72,72,79},
     {62,66,69,74,74,81}, {64,68,71,75,76,83}, {65,67,72,77,77,84},
     {67,71,74,79,79,86}
 };
 static const std::vector<std::string> MODES = {"rev_arp","random","forward_arp"};
 
 // Frequency Lookup Table
 static double freqLUT[128];
 static inline void initFreqLUT() {
     for(int i = 0; i < 128; ++i) {
         freqLUT[i] = 440.0 * std::pow(2.0, (i - 69) / 12.0);
     }
 }
 
 // Shared state
 std::atomic<bool> running{true};
 std::deque<std::vector<float>> audioQueue;
 std::mutex seqMutex, wavMutex, featMutex;
 std::vector<int> globalSeq;
 std::string    globalWaveform = "sine";
 float          globalBrightness = 0.0f;
 float          globalWarmth     = 0.0f;
 float          globalTexture    = 0.0f;
 int            globalObjCount  = 0;
 float          globalReverbMix  = 1.0f;
 std::vector<float> globalReverbTail(DLY_FRAMES, 0.0f);
 std::mt19937 rng{std::random_device{}()};
 
 // Utility functions
 inline double remap(double v,double a,double b,double c,double d){ return (v - a)/(b - a)*(d - c) + c; }
 
 // Wave generator
 std::vector<float> generateWave(double freq,int len,float amp,const std::string &kind){
     std::vector<float> out(len);
     for(int i=0;i<len;++i){
         double t = i / static_cast<double>(FS);
         double phase = 2*M_PI * freq * t;
         double v = 0;
         if(kind=="sine")       v = std::sin(phase);
         else if(kind=="triangle"){ double frac = std::fmod(t*freq,1.0); v = 2*std::abs(2*(frac-0.5))-1; }
         else if(kind=="square") v = (std::sin(phase) >= 0? 1.0: -1.0);
         out[i] = float(amp * v);
     }
     // 5ms fade-in/out to avoid clicks <<< TUNABLE fade_ms >>>
     int fade = int(0.005 * FS);
     for(int i=0;i<fade && i<len; ++i){ float g = float(i)/fade; out[i]*=g; out[len-1-i]*=g; }
     return out;
 }
 
 // Feature extraction thread (matches Python logic)
 void featureThread(){
     VideoCapture cap(0);
     cap.set(CAP_PROP_FRAME_WIDTH,160);
     cap.set(CAP_PROP_FRAME_HEIGHT,120);
     Mat frame, small, gray, edges;
     while(running){
         auto t0 = high_resolution_clock::now();
         if(!cap.read(frame)){ std::this_thread::sleep_for(milliseconds(50)); continue; }
         cv::resize(frame, small, Size(160,120));
         cvtColor(small, gray, COLOR_BGR2GRAY);
         float brightness = mean(gray)[0]/255.0f;
         auto bgr = std::vector<Mat>{}; split(small, bgr);
         float r = mean(bgr[2])[0]; float g = mean(bgr[1])[0]; float b = mean(bgr[0])[0];
         float warmth = remap((2*r)/(2*(b+g)+1), 0.45,0.55, 0, static_cast<double>(CHORDS.size()) - 0.001);
         cv::Canny(gray, edges, 50,150);
         float texture = float(countNonZero(edges))/edges.total();
         Mat thr; adaptiveThreshold(gray, thr,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,11,2);
         Mat clean; morphologyEx(thr, clean, MORPH_OPEN, Mat::ones(3,3,CV_8U),Point(-1,-1),2);
         std::vector<std::vector<Point>> contours;
         findContours(clean, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
         int objCount = (int)contours.size();
         { std::lock_guard<std::mutex> lk(featMutex);
             globalBrightness = brightness;
             globalWarmth     = warmth;
             globalTexture    = texture;
             globalObjCount   = objCount;
         }
         std::this_thread::sleep_until(t0 + milliseconds(50));
     }
 }
 
 // Chord updater thread
 void chordThread(){
     std::uniform_int_distribution<int> modeDist(0,2);
     while(running){
         float warmth, brightness;
         { std::lock_guard<std::mutex> lk(featMutex); warmth=globalWarmth; brightness=globalBrightness; }
         int idx = std::min<int>(int(warmth), CHORDS.size()-1);
         auto seq = CHORDS[idx];
         int mode = modeDist(rng);
         if(mode==0) std::reverse(seq.begin(), seq.end());
         else if(mode==1) std::shuffle(seq.begin(), seq.end(), rng);
         else if(mode==2 && !seq.empty()){ seq.push_back(seq.front()); seq.erase(seq.begin()); }
         int offs = int(std::floor(remap(brightness,0,1,-4,3))) * 12;
         for(auto &n: seq) n += offs;
         seq.push_back(seq.front());
         { std::lock_guard<std::mutex> lk(seqMutex); globalSeq = std::move(seq); }
         std::this_thread::sleep_for(milliseconds(int(DURATION*1000)));
     }
 }
 
 // Waveform updater thread
 void waveformThread(){
     while(running){
         float texture;
         { std::lock_guard<std::mutex> lk(featMutex); texture=globalTexture; }
         std::string wf = (texture<0.05f? "sine": (texture<0.1f? "triangle": "square"));
         { std::lock_guard<std::mutex> lk(wavMutex); globalWaveform=wf; }
         std::this_thread::sleep_for(milliseconds(int(SUB*1000)));
     }
 }
 
 // Reverb updater thread
 void reverbThread(){
     while(running){
         int objCount;
         { std::lock_guard<std::mutex> lk(featMutex); objCount=globalObjCount; }
         float ratio = std::min(objCount/25.0f,1.0f);
         float sens = ratio*ratio;
         float rv = float(remap(sens,0,1,0.9,0.1));
         globalReverbMix = std::clamp(rv, 0.0f, 1.0f);
         std::this_thread::sleep_for(seconds(1));
     }
 }
 
 // Player thread: schedules chunks into audioQueue
 void playerThread(){
     size_t idx=0;
     while(running){
         std::vector<int> seq;
         std::string wf;
         float texture, volume, revMix;
         { std::lock_guard<std::mutex> lk1(seqMutex); seq=globalSeq; }
         { std::lock_guard<std::mutex> lk2(wavMutex); wf=globalWaveform; }
         { std::lock_guard<std::mutex> lk3(featMutex); texture=globalTexture; }
         volume = 1.0f;
         revMix = globalReverbMix;
         if(!seq.empty()){
             int note = seq[idx++ % seq.size()];
             double freq = freqLUT[note];  // use LUT
             auto dry = generateWave(freq, SUB_FRAMES, volume, wf);
             int rel = int((1 - texture*texture) * dry.size());
             for(int i=0;i<rel;++i) dry[dry.size()-1-i] *= float(i)/rel;
             std::vector<float> out(dry.size());
             for(size_t i=0;i<dry.size(); ++i){
                 float old = globalReverbTail[i % DLY_FRAMES];
                 float newEcho = (i<dry.size()-DLY_FRAMES? dry[i] : 0.0f) * revMix;
                 float mix = dry[i] * (1-revMix) + old + newEcho;
                 out[i] = mix;
                 globalReverbTail[i % DLY_FRAMES] = mix;
             }
             { std::lock_guard<std::mutex> lkQ(seqMutex);
                 audioQueue.push_back(out);
                 if(audioQueue.size()>10) audioQueue.pop_front();
             }
         }
         std::this_thread::sleep_for(milliseconds(int(SUB*1000)));
     }
 }
 
 // ALSA audio thread using MMAP
 void audioThread(){
     snd_pcm_t *pcm;
     snd_pcm_hw_params_t *hw;
     snd_pcm_sw_params_t *sw;
 
     snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0);
     snd_pcm_hw_params_malloc(&hw);
     snd_pcm_hw_params_any(pcm, hw);
     snd_pcm_hw_params_set_access(pcm, hw, ACCESS);
     snd_pcm_hw_params_set_format(pcm, hw, FORMAT);
     snd_pcm_hw_params_set_channels(pcm, hw, CHANNELS);
     unsigned int rate = FS;
     snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, nullptr);
     snd_pcm_uframes_t period = PERIOD_FRAMES;
     snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, nullptr);
     snd_pcm_uframes_t bufsize = PERIOD_FRAMES * BUFFER_PERIODS;
     snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &bufsize);
     snd_pcm_hw_params(pcm, hw);
     snd_pcm_hw_params_free(hw);
 
     snd_pcm_sw_params_malloc(&sw);
     snd_pcm_sw_params_current(pcm, sw);
     snd_pcm_sw_params_set_start_threshold(pcm, sw, period);
     snd_pcm_sw_params_set_avail_min(pcm, sw, period);
     snd_pcm_sw_params(pcm, sw);
     snd_pcm_sw_params_free(sw);
 
     snd_pcm_prepare(pcm);
 
     const snd_pcm_channel_area_t *areas;
     snd_pcm_uframes_t offset, frames;
     while(running){
         snd_pcm_sframes_t avail = snd_pcm_avail_update(pcm);
         if(avail < 0) { snd_pcm_prepare(pcm); continue; }
         while(avail >= (snd_pcm_sframes_t)PERIOD_FRAMES){
             frames = PERIOD_FRAMES;
             snd_pcm_mmap_begin(pcm, &areas, &offset, &frames);
             std::vector<float> chunk;
             {
                 std::lock_guard<std::mutex> lk(seqMutex);
                 if(!audioQueue.empty()){ chunk = audioQueue.front(); audioQueue.pop_front(); }
             }
             auto ptr = reinterpret_cast<int16_t*>(areas[0].addr) + offset;
             for(snd_pcm_uframes_t i=0; i<frames; ++i){
                 int16_t sample = 0;
                 if(i < chunk.size()) sample = int16_t(std::clamp(chunk[i], -1.0f, 1.0f)*32767);
                 ptr[i] = sample;
             }
             snd_pcm_mmap_commit(pcm, offset, frames);
             avail = snd_pcm_avail_update(pcm);
         }
         std::this_thread::sleep_for(milliseconds(1));
     }
 
     snd_pcm_close(pcm);
 }
 
 int main(){
     initFreqLUT();              // initialize LUT <<< ADDED >>>
     globalSeq = CHORDS[0];
     std::thread tFeat(featureThread);
     std::thread tChord(chordThread);
     std::thread tWav(waveformThread);
     std::thread tRev(reverbThread);
     std::thread tPlay(playerThread);
     std::thread tAudio(audioThread);
 
     std::cout<<"[INFO] ALSA Synth running. Press Enter to quit."<<std::endl;
     std::cin.get(); running=false;
 
     tFeat.join(); tChord.join(); tWav.join(); tRev.join(); tPlay.join(); tAudio.join();
     return 0;
 }
 