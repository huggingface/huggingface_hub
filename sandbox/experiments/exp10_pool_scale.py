# /// script
# dependencies = ["huggingface_hub"]
# ///
"""Scale benchmark for SandboxPool: N landlock sandboxes packed across host VMs.

Measures, for N sandboxes packed `--per-host` per host job on `--flavor`:
  1. provision + create  — boot ceil(N/per_host) hosts (in parallel) and create
     all N sandboxes (one batched request per host),
  2. exec                — run `echo` in every sandbox (parallel over the proxy),
  3. kill                — tear everything down (cancel all host jobs).

Usage:
    python exp10_pool_scale.py                       # N=1000, per_host=50, cpu-basic
    python exp10_pool_scale.py --num 200 --per-host 50 --flavor cpu-basic
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import SandboxPool
from huggingface_hub._sandbox import CommandResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--per-host", type=int, default=50)
    parser.add_argument("--flavor", type=str, default="cpu-basic")
    parser.add_argument("--exec-workers", type=int, default=200)
    args = parser.parse_args()

    n, per_host = args.num, args.per_host
    num_hosts = -(-n // per_host)
    print(f"target: {n} sandboxes, {per_host}/host -> {num_hosts} host(s) on {args.flavor}\n")

    pool = SandboxPool(
        image="python:3.12",
        flavor=args.flavor,
        sandboxes_per_host=per_host,
        timeout="30m",
    )
    try:
        # 1. provision hosts + create all sandboxes
        t0 = time.perf_counter()
        boxes = pool.create(count=n)
        t1 = time.perf_counter()
        print(f"[1] provision {pool.num_hosts} host(s) + create {len(boxes)} sandboxes: {t1 - t0:6.1f}s")

        # 2. exec echo in every sandbox, in parallel over the proxy
        def echo(box) -> tuple[bool, float, int]:
            # One retry: the HF Jobs proxy occasionally drops a connection under heavy
            # fan-out (transient RemoteProtocolError), unrelated to the sandbox itself.
            s = time.perf_counter()
            retries = 0
            for attempt in range(2):
                try:
                    r = box.run("echo hello", check=False)
                    if isinstance(r, CommandResult) and r.stdout == "hello\n":
                        return True, time.perf_counter() - s, retries
                except Exception:
                    pass
                retries = attempt + 1
            return False, time.perf_counter() - s, retries

        t2 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.exec_workers) as pool_ex:
            results = list(pool_ex.map(echo, boxes))
        t3 = time.perf_counter()
        ok = sum(1 for r, _, _ in results if r)
        retried = sum(1 for _, _, rt in results if rt)
        latencies = sorted(1000 * d for _, d, _ in results)
        print(
            f"[2] exec echo in all {len(boxes)} (≤{args.exec_workers} concurrent): {t3 - t2:6.1f}s  "
            f"({ok}/{len(boxes)} ok, {retried} needed a proxy retry)"
        )
        print(
            f"    per-exec latency: p50={statistics.median(latencies):.0f}ms "
            f"p95={latencies[int(0.95 * len(latencies))]:.0f}ms max={latencies[-1]:.0f}ms"
        )

        # 3. teardown
        t4 = time.perf_counter()
        pool.close()
        t5 = time.perf_counter()
        print(f"[3] kill {num_hosts} host(s) (all sandboxes):              {t5 - t4:6.1f}s")
        print(f"\nTOTAL create+exec+kill: {t5 - t0:.1f}s  for {n} sandboxes across {num_hosts} hosts")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
