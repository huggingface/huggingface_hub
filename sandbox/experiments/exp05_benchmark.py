"""Experiment 5: comprehensive benchmark of the sandbox stack on real jobs.

Measures: cold start (variants), exec latency, parallel exec, streaming liveness,
file throughput (vs raw http.server through the same proxy).
"""

import concurrent.futures
import os
import statistics
import time

from huggingface_hub import Sandbox


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]


results: dict[str, str] = {}

# ---------------------------------------------------------------- cold start
print("=== cold start: 5x python:3.12 (download bootstrap) ===")
cold = []
for i in range(5):
    t0 = time.time()
    sbx = Sandbox.create(timeout="10m")
    dt = time.time() - t0
    cold.append(dt)
    print(f"  run {i}: {dt:.1f}s")
    if i < 4:
        sbx.kill()
results["cold_python312"] = f"median {statistics.median(cold):.1f}s (min {min(cold):.1f}, max {max(cold):.1f})"

print("=== cold start: alpine:3.20 ===")
t0 = time.time()
alpine = Sandbox.create(image="alpine:3.20", timeout="10m")
results["cold_alpine"] = f"{time.time() - t0:.1f}s"
print(f"  {results['cold_alpine']}")
alpine.kill()

print("=== cold start: mount mode ===")
t0 = time.time()
mnt = Sandbox.create(image="alpine:3.20", server_source="mount", timeout="10m")
results["cold_mount"] = f"{time.time() - t0:.1f}s"
print(f"  {results['cold_mount']}")
mnt.kill()

print("=== cold start: 4 sandboxes in parallel ===")
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    sandboxes = list(pool.map(lambda _: Sandbox.create(timeout="10m"), range(4)))
results["cold_parallel_4"] = f"{time.time() - t0:.1f}s wall for 4"
print(f"  {results['cold_parallel_4']}")
for s in sandboxes:
    s.kill()

# ---------------------------------------------------------------- exec on the kept sandbox
# `sbx` is the 5th sandbox from the cold-start loop, still alive
print(f"=== exec benchmarks on {sbx.id} ===")

lat = []
for _ in range(30):
    t = time.time()
    sbx.run("true")
    lat.append((time.time() - t) * 1000)
results["exec_latency"] = f"p50 {pct(lat, 50):.0f}ms / p90 {pct(lat, 90):.0f}ms / min {min(lat):.0f}ms"
print(f"  sequential: {results['exec_latency']}")

# parallel exec: 16 concurrent commands, each sleeping 1s
t0 = time.time()


def timed_run(i):
    # NOTE: requests.Session is documented as not thread-safe; create one Sandbox
    # connection per thread via connect() to be safe.
    s = Sandbox.connect(sbx.id)
    return s.run("sleep 1 && echo done").stdout.strip()


with concurrent.futures.ThreadPoolExecutor(16) as pool:
    outs = list(pool.map(timed_run, range(16)))
wall = time.time() - t0
assert all(o == "done" for o in outs)
results["exec_parallel_16"] = f"16x 'sleep 1' in {wall:.1f}s wall (ideal ~1s + overhead)"
print(f"  parallel: {results['exec_parallel_16']}")

# streaming liveness through the proxy
arrivals = []
t0 = time.time()
sbx.run(
    "for i in 1 2 3 4 5 6; do echo tick; sleep 0.3; done",
    on_stdout=lambda d: arrivals.append(time.time() - t0),
)
gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
results["stream_liveness"] = (
    f"6 ticks @300ms: gaps median {statistics.median(gaps) * 1000:.0f}ms, max {max(gaps) * 1000:.0f}ms"
)
print(f"  streaming: {results['stream_liveness']}")

# ---------------------------------------------------------------- file throughput
print("=== file throughput (32 MiB) through proxy ===")
blob = os.urandom(32 * 1024 * 1024)
t = time.time()
sbx.files.write("/tmp/blob.bin", blob)
up = 32 / (time.time() - t)
t = time.time()
back = sbx.files.read("/tmp/blob.bin")
down = 32 / (time.time() - t)
assert back == blob
results["file_throughput"] = f"upload {up:.1f} MiB/s, download {down:.1f} MiB/s"
print(f"  sbx-server: {results['file_throughput']}")

# comparison: python http.server through the same proxy (is the proxy the limit?)
sbx.spawn("cd /tmp && python -m http.server 8080", tag="web")
time.sleep(1.0)
# in-sandbox loopback download as an upper bound (proxy out of the picture)
r = sbx.run("python -c \"import time,urllib.request; t=time.time(); d=urllib.request.urlopen('http://localhost:8080/blob.bin').read(); print(f'{32/(time.time()-t):.0f}')\"")
results["loopback_inside"] = f"{r.stdout.strip()} MiB/s (in-sandbox http.server loopback)"
print(f"  in-sandbox loopback: {results['loopback_inside']}")

sbx.kill()

print("\n=== SUMMARY ===")
for k, v in results.items():
    print(f"  {k}: {v}")
