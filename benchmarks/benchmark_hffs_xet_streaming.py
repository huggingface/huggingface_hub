"""Benchmark HfFileSystem streaming reads: xet vs HTTP.

Run twice — once with HF_HUB_DISABLE_XET=1 (HTTP baseline) and once without (xet) —
and compare. Pass a real xet-backed hf:// file path (parquet or any large file).

Usage:
    HF_HUB_DISABLE_XET=1 python benchmarks/benchmark_hffs_xet_streaming.py <hf://...>
    python benchmarks/benchmark_hffs_xet_streaming.py <hf://...>
"""

import sys
import time

from huggingface_hub import HfFileSystem
from huggingface_hub.utils._runtime import is_xet_available


def _time(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main(path: str) -> None:
    fs = HfFileSystem()
    size = fs.info(path)["size"]
    mode = "xet" if is_xet_available() else "http"
    print(f"mode={mode} size={size} path={path}")

    def footer_read():
        with fs.open(path, "rb") as f:
            f.seek(max(0, size - 64 * 1024))
            f.read(64 * 1024)

    def full_scan():
        with fs.open(path, "rb", block_size=0) as f:
            while f.read(8 * 1024 * 1024):
                pass

    def random_access():
        with fs.open(path, "rb") as f:
            step = max(1, size // 16)
            for off in range(0, size, step):
                f.seek(off)
                f.read(min(256 * 1024, size - off))

    print(f"footer_read     : {_time(footer_read):.3f}s")
    print(f"full_scan       : {_time(full_scan):.3f}s")
    print(f"random_access   : {_time(random_access):.3f}s")


if __name__ == "__main__":
    main(sys.argv[1])
