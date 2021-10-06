#!/usr/bin/env bash


THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MACHINE_SETUP_DIR=${THIS_DIR}/machine_setup

${MACHINE_SETUP_DIR}/disable_hyper_threading.sh
${MACHINE_SETUP_DIR}/disable_turbo_boost.sh
${MACHINE_SETUP_DIR}/drop_caches.sh
${MACHINE_SETUP_DIR}/randomize_va_space.sh
${MACHINE_SETUP_DIR}/scaling_governor.sh