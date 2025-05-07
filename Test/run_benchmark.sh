#!/bin/sh
cd /root

# cleanup function to kill loggers and list logs
cleanup() {
  kill $pid_mem $pid_alsa $pid_fps 2>/dev/null
  echo
  echo "Finished. Logs are:"
  ls -1 full_synth_opt_min.log mem_load_opt_min.log alsa_opt_min.log fps_opt_min.log
  exit 0
}

# trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup INT TERM

# 0) remove old logs
rm -f full_synth_opt_min.log mem_load_opt_min.log alsa_opt_min.log fps_opt_min.log vis_fps_opt_min.log

# 1) start mem+load logger
(
  echo "TIME(s) PID MEM_KB LOAD1 LOAD5 LOAD15"
  while :; do
    pid=$(pidof full_synth) || break
    t=$(date +%s)
    read _ res _ < /proc/$pid/statm
    mem_kb=$((res * 4))
    read load1 load5 load15 _ < /proc/loadavg
    echo "$t $pid $mem_kb $load1 $load5 $load15"
    sleep 1
  done
) > mem_load_opt_min.log 2>&1 &
pid_mem=$!

# 2) enhanced ALSA underrun logger
(
  echo "TIME(s) UNDERRUNS"
  # wait until the PCM enters RUNNING state
  until grep -q '^state:.*RUNNING' /proc/asound/card3/pcm0p/sub0/status 2>/dev/null; do
    sleep 0.1
  done
  # now poll underruns once per second while still RUNNING
  while grep -q '^state:.*RUNNING' /proc/asound/card3/pcm0p/sub0/status; do
    t=$(date +%s)
    u=$(grep -i '^underruns:' /proc/asound/card3/pcm0p/sub0/status \
         | awk '{print $2}')
    [ -z "$u" ] && u=0
    echo "$t $u"
    sleep 1
  done
) > alsa_opt_min.log 2>&1 &
pid_alsa=$!

# 3) start audio-fps logger
(
  echo "TIME(s) AudioPeriodRate"
  lastfps=0
  while :; do
    pid=$(pidof full_synth) || break
    t=$(date +%s)
    total=$(grep -c "\[FEEDBACK_LOG\]" full_synth_opt_min.log)
    fps=$(( total - lastfps ))
    lastfps=$total
    echo "$t $fps"
    sleep 1
  done
) > fps_opt_min.log 2>&1 &
pid_fps=$!

# 3b) start VISION-FPS logger (counts [WARMTH_DBG] per second)
(
  echo "TIME(s) FPS"
  lastvfps=0
  while :; do
    pid=$(pidof full_synth) || break
    t=$(date +%s)
    totalvfps=$(grep -c "\[WARMTH_DBG\]" full_synth_opt_min.log)
    vfps=$((totalvfps - lastvfps))
    lastvfps=$totalvfps
    echo "$t $vfps"
    sleep 1
  done
) > vis_fps_opt_min.log 2>&1 &
pid_vis=$!

# 4) run the synth in foreground, tee’ing output to its log
./full_synth 2>&1 | tee full_synth_opt_min.log

# 5) if it exits normally (or via Ctrl+C), clean up
cleanup
