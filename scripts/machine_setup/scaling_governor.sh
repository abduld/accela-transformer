#!/usr/bin/env bash

sudo apt-get install cpufrequtils
sudo cpufreq-set -r -g performance

for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $i
done
