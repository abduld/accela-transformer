#!/usr/bin/env bash

# sudo sh -c "mkdir -p /sys/devices/system/cpu/intel_pstate"
# sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"

sudo turbostat stress -c 2 -t 10
