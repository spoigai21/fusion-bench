#!/usr/bin/env bash
# Phase 0 — GPU access and counter access.
#
# Run this FIRST, before anything else. The entire project rests on Nsight Compute
# being able to read hardware counters, and many cloud containers (and every Colab
# runtime) block them. Finding that out after writing three kernels is the one
# genuinely expensive mistake available here.
#
#   bash scripts/phase0_check.sh
#
# If the counter test fails with a permissions error, try in order:
#   1. restart the container with --cap-add=SYS_ADMIN
#   2. a different provider
#   3. a bare-metal instance
# If none of those work, the project switches to Plan B (derived bytes, see
# bench/parse_ncu.py --analytic) and the README must say so plainly.

set -u

pass=0
fail=0

step() { printf '\n=== %s\n' "$1"; }
ok()   { printf '  [ok]   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=$((fail + 1)); }

step "1. GPU visible"
if nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
  ok "nvidia-smi"
else
  bad "nvidia-smi — no GPU visible to this container"
fi

step "2. compiler present"
if nvcc --version | tail -2; then ok "nvcc"; else bad "nvcc not on PATH"; fi

step "3. profiler present"
if ncu --version | head -3; then ok "ncu"; else bad "ncu not installed"; fi

step "4. torch sees the device"
if python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"; then
  ok "torch"
else
  bad "torch cannot see a CUDA device"
fi

step "5. THE TEST THAT MATTERS — do counters come back?"
# Profile anything at all and confirm a byte count appears rather than a permissions
# error. This is the phase 0 checkpoint; do not proceed without it.
tmp=$(mktemp)
if ncu --metrics dram__bytes_read.sum --print-units base --csv \
      python -c "import torch; x=torch.randn(4096,4096,device='cuda'); torch.softmax(x,-1); torch.cuda.synchronize()" \
      >"$tmp" 2>&1; then
  if grep -q "dram__bytes_read" "$tmp"; then
    ok "counters readable"
    grep -m3 "dram__bytes_read" "$tmp"
  else
    bad "ncu ran but produced no dram__bytes_read row"
    tail -20 "$tmp"
  fi
else
  bad "ncu failed — see output below"
  tail -30 "$tmp"
  cat <<'HINT'

  ERR_NVGPUCTRPERM means the driver is restricting counters to admin users
  (NVreg_RestrictProfilingToAdminUsers=1, the default). In order:

    1. If you have root -- a bare VM or bare metal -- just use sudo:
         sudo $(command -v ncu) --metrics dram__bytes_read.sum ... python3 ...
       and run the project's profiling with NCU_SUDO=1 bash scripts/profile.sh
       This is what worked on Lambda and needs no reboot.
    2. In a container: restart it with --cap-add=SYS_ADMIN (RunPod: privileged mode).
    3. A different provider, or bare metal.
    4. Plan B: python bench/parse_ncu.py --analytic   (and say so in the README)

  A stale /tmp/nsight-compute-lock from a failed run blocks even root; delete it.
HINT
fi
rm -f "$tmp"

step "6. metric names on THIS ncu version"
# Metric names move between Nsight versions; verify rather than trusting a copied
# command line.
for m in dram__bytes_read.sum dram__bytes_write.sum lts__t_bytes.sum; do
  if ncu --query-metrics 2>/dev/null | grep -q "${m%%.*}"; then
    ok "$m available"
  else
    bad "$m NOT found by --query-metrics — check the name on this version"
  fi
done

step "7. can clocks be locked?"
if nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv,noheader; then
  echo "  to lock (needs root):  bash scripts/lock_clocks.sh"
  echo "  if you cannot lock them, that is fine — the harness logs achieved clocks per run"
fi

printf '\n=== phase 0: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
