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
#
# lts__t_bytes.sum (L2 traffic) is captured alongside the DRAM metrics on purpose: at
# 4096x1024 the working set fits in the A100's 40 MB L2, so v0's extra passes hit L2
# rather than DRAM. Without the L2 counter that effect can only be speculated about.
set -eu

cd "$(dirname "$0")/.."

SHAPES=${*:-"4096x1024 4096x8192"}
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum"
EXTRA=""
[ "${WITH_TORCH:-0}" = "1" ] && EXTRA="--with-torch"

mkdir -p results/raw

# Compile the extension outside the profiler; a ~60 s nvcc run under ncu is just noise.
echo "warming the build cache..."
python -c "import bench.ext" >/dev/null

for shape in $SHAPES; do
  out="results/raw/ncu_${shape}.csv"
  echo "profiling $shape -> $out"
  ncu --metrics "$METRICS" \
      --print-units base \
      --csv \
      python bench/profile_one.py --shape "$shape" $EXTRA > "$out"
  echo "  $(grep -c . "$out") lines"
done

echo
echo "now: python bench/parse_ncu.py    # -> results/bytes.csv"
