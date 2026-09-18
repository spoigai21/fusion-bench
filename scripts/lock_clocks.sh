#!/usr/bin/env bash
# Lock clocks so the timing distribution reflects the kernel, not the clock governor.
# Needs root, and many rented containers will refuse. That is survivable: the harness
# records achieved SM clock on every run either way, and the median over 200 launches
# is robust to occasional boost changes.
#
#   sudo bash scripts/lock_clocks.sh          # lock to the A100 base clocks
#   sudo bash scripts/lock_clocks.sh reset
set -eu

if [ "${1:-lock}" = "reset" ]; then
  nvidia-smi --reset-gpu-clocks
  nvidia-smi --reset-memory-clocks
  echo "clocks reset"
  exit 0
fi

SM_CLOCK="${SM_CLOCK:-1095}"
MEM_CLOCK="${MEM_CLOCK:-1215}"

nvidia-smi -pm 1 || true
nvidia-smi --lock-gpu-clocks="$SM_CLOCK","$SM_CLOCK"
nvidia-smi --lock-memory-clocks="$MEM_CLOCK","$MEM_CLOCK"
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader
echo "locked — remember to run 'sudo bash scripts/lock_clocks.sh reset' before releasing the box"
