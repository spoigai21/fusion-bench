#!/usr/bin/env bash
# Phase 6 — the byte table.
#
# ncu instruments every launch it sees, so this points at bench/profile_one.py (exactly
# one launch per kernel) and never at the benchmark loop. Output lands in
# results/raw/ncu_<N>x<D>.csv and is turned into results/bytes.csv by bench/parse_ncu.py.
#
#   bash scripts/profile.sh                       # both shapes
#   bash scripts/profile.sh 4096x8192             # one shape
#   WITH_TORCH=1 bash scripts/profile.sh          # include torch.softmax's counters
#   WITH_VARIANTS=1 bash scripts/profile.sh       # also the v2a/v2b/v2c byte table
#   NCU_SUDO=1 bash scripts/profile.sh            # when counters need root
#
# On ERR_NVGPUCTRPERM: the driver restricts counters to admin users by default
# (NVreg_RestrictProfilingToAdminUsers=1). In a container the fix is --cap-add=SYS_ADMIN.
# On a bare VM where you have root -- Lambda, most bare metal -- NCU_SUDO=1 is enough,
# and is what this project used. Root gets its own extension build directory so it never
# leaves root-owned objects in the .build/ the normal user writes to.
#
# lts__t_bytes.sum (L2 traffic) is captured alongside the DRAM metrics on purpose: at
# 4096x1024 the working set fits in the A100's 40 MB L2, so v0's extra passes hit L2
# rather than DRAM. Without the L2 counter that effect can only be speculated about.
set -eu

cd "$(dirname "$0")/.."

SHAPES=${*:-"4096x1024 4096x8192"}
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum"

NCU_BIN="${NCU_BIN:-$(command -v ncu || echo ncu)}"
if [ "${NCU_SUDO:-0}" = "1" ]; then
  NCU_RUN="sudo FUSION_BENCH_BUILD_DIR=${ROOT_BUILD_DIR:-/tmp/fusion-bench-build-root} $NCU_BIN"
  # A stale lock from an earlier failed run blocks even root.
  sudo rm -f /tmp/nsight-compute-lock
else
  NCU_RUN="$NCU_BIN"
fi
EXTRA=""
[ "${WITH_TORCH:-0}" = "1" ] && EXTRA="--with-torch"

mkdir -p results/raw

# Compile the extension outside the profiler; a ~60 s nvcc run under ncu is just noise.
echo "warming the build cache..."
python3 -c "import bench.ext" >/dev/null

for shape in $SHAPES; do
  out="results/raw/ncu_${shape}.csv"
  echo "profiling $shape -> $out"
  $NCU_RUN --metrics "$METRICS" \
      --print-units base \
      --csv \
      python3 bench/profile_one.py --shape "$shape" $EXTRA > "$out"
  echo "  $(grep -c . "$out") lines"

  # The attribution ladder goes in a SEPARATE run: v2 dispatches to the same CUDA
  # kernels as v2b and v2c, so profiling them together would leave ncu rows that cannot
  # be told apart by kernel name. The filename is what tells parse_ncu.py which mapping
  # to use, so keep the ncu_variants_ prefix.
  if [ "${WITH_VARIANTS:-0}" = "1" ]; then
    vout="results/raw/ncu_variants_${shape}.csv"
    echo "profiling $shape variants -> $vout"
    $NCU_RUN --metrics "$METRICS" \
        --print-units base \
        --csv \
        python3 bench/profile_one.py --shape "$shape" --kernels v2a,v2b,v2c > "$vout"
    echo "  $(grep -c . "$vout") lines"
  fi
done

echo
echo "now: python bench/parse_ncu.py    # -> results/bytes.csv"
