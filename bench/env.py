"""Environment capture. A number without its environment is not reproducible.

Every generated CSV carries this block as `#` comment lines above the header, and
results/env.json holds the same thing in machine-readable form.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Stated peak memory bandwidth, GB/s. This is the denominator for every "% of peak"
# figure in the write-up, so the exact SKU matters: an A100 80GB PCIe and an A100 80GB
# SXM4 differ by 5%. Keys are matched as substrings of torch's device name, longest
# first. Override with PEAK_BW_GBS=<number> for a card that is not listed.
PEAK_BW_GBS = {
    "A100-SXM4-80GB": 2039.0,
    "A100-PCIE-80GB": 1935.0,
    "A100 80GB PCIe": 1935.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-PCIE-40GB": 1555.0,
    "H100 PCIe": 2000.0,
    "H100 80GB HBM3": 3350.0,
    "H200": 4800.0,
    "L40S": 864.0,
    "RTX 6000 Ada": 960.0,
    "RTX A6000": 768.0,
    "RTX 4090": 1008.0,
    "RTX 3090": 936.0,
    "V100-SXM2-32GB": 900.0,
    "T4": 320.0,
}


def _sh(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def peak_bandwidth_gbs(device_name: str) -> tuple[float | None, str]:
    """Return (GB/s, source). None means unknown -- % of peak is then left blank."""
    override = os.environ.get("PEAK_BW_GBS")
    if override:
        return float(override), "PEAK_BW_GBS env override"
    for key in sorted(PEAK_BW_GBS, key=len, reverse=True):
        if key.lower().replace(" ", "") in device_name.lower().replace(" ", ""):
            return PEAK_BW_GBS[key], f"table lookup: {key}"
    return None, "unknown SKU -- set PEAK_BW_GBS to enable %-of-peak"


def nvidia_smi_clocks() -> dict[str, str]:
    """Achieved clocks right now. Logged whether or not clocks are locked."""
    q = _sh(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
            "clocks_throttle_reasons.active,temperature.gpu,power.draw",
            "--format=csv,noheader",
            "-i",
            "0",
        ]
    )
    if not q:
        return {}
    parts = [p.strip() for p in q.split(",")]
    keys = [
        "sm_clock",
        "sm_clock_max",
        "mem_clock",
        "mem_clock_max",
        "throttle_reasons",
        "temp_c",
        "power_w",
    ]
    return dict(zip(keys, parts))


def git_commit() -> str:
    """Short SHA, suffixed -dirty if the tree has uncommitted changes.

    An empty repo and a non-repo are reported differently: a CSV that says
    no-commits-yet is telling you the numbers in it cannot be tied to a tree state,
    which is a different problem from not using git at all.
    """
    if _sh(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"]) != "true":
        return "not-a-git-repo"
    sha = _sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    if not sha:
        return "no-commits-yet"
    dirty = _sh(["git", "-C", str(ROOT), "status", "--porcelain"])
    return sha + ("-dirty" if dirty else "")


def collect() -> dict:
    """Everything the write-up needs to be reproducible."""
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": git_commit(),
        "nvcc": (_sh(["nvcc", "--version"]).splitlines() or [""])[-1],
        "ncu": (_sh(["ncu", "--version"]).splitlines() or [""])[-1],
        "driver": _sh(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"]
        ),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda or ""
        info["cudnn"] = str(torch.backends.cudnn.version())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            info["sm_count"] = props.multi_processor_count
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["shared_mem_per_block_optin_kb"] = round(
                getattr(props, "shared_memory_per_block_optin", 0) / 1024, 1
            )
            bw, src = peak_bandwidth_gbs(props.name)
            info["peak_bw_gbs"] = bw
            info["peak_bw_source"] = src
        else:
            info["gpu"] = "none (CUDA unavailable)"
    except Exception as exc:  # torch missing entirely
        info["torch"] = f"unavailable: {exc}"

    info.update({f"clocks_{k}": v for k, v in nvidia_smi_clocks().items()})
    return info


def header_lines(info: dict) -> list[str]:
    """The `#` comment block written above every CSV header."""
    return [f"# {k}={v}" for k, v in info.items()]


def write_env_json(path: Path | None = None) -> Path:
    info = collect()
    path = path or (ROOT / "results" / "env.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
