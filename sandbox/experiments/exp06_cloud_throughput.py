"""Experiment 6: proxy file throughput measured cloud-to-cloud (local uplink is ~2MB/s,
so sandbox B benchmarks against sandbox A's public URL from inside the cloud)."""

import time

from huggingface_hub import Sandbox


BENCH = r"""
import concurrent.futures, json, os, time, urllib.request

BASE = os.environ["A_URL"]
HEADERS = {"Authorization": "Bearer " + os.environ["A_HF_TOKEN"], "X-Sandbox-Token": os.environ["A_SBX_TOKEN"]}
SIZE = 32 * 1024 * 1024
CHUNK = 4 * 1024 * 1024

def get(path, params=""):
    req = urllib.request.Request(BASE + path + ("?" + params if params else ""), headers=HEADERS)
    return urllib.request.urlopen(req, timeout=120).read()

def put(path, params, data):
    req = urllib.request.Request(BASE + path + "?" + params, data=data, headers=HEADERS, method="PUT")
    return urllib.request.urlopen(req, timeout=120).read()

# single-stream download
t = time.time(); d = get("/v1/files/read", "path=/tmp/blob.bin"); assert len(d) == SIZE
print(f"download single: {SIZE/1024/1024/(time.time()-t):.1f} MiB/s")

# parallel download (8 ranged)
def fetch(off):
    return get("/v1/files/read", f"path=/tmp/blob.bin&offset={off}&length={CHUNK}")
t = time.time()
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    parts = list(pool.map(fetch, range(0, SIZE, CHUNK)))
assert sum(len(p) for p in parts) == SIZE
print(f"download parallel8: {SIZE/1024/1024/(time.time()-t):.1f} MiB/s")

# single-stream upload
blob = os.urandom(SIZE)
t = time.time(); put("/v1/files/write", "path=/tmp/up1.bin", blob)
print(f"upload single: {SIZE/1024/1024/(time.time()-t):.1f} MiB/s")

# parallel upload
def push(off):
    put("/v1/files/write", f"path=/tmp/up2.bin&offset={off}", blob[off:off+CHUNK])
t = time.time()
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    list(pool.map(push, range(0, SIZE, CHUNK)))
print(f"upload parallel8: {SIZE/1024/1024/(time.time()-t):.1f} MiB/s")
"""

a = Sandbox.create(timeout="10m")
print("A:", a.id)
b = Sandbox.create(timeout="10m")
print("B:", b.id)
try:
    a.run("head -c 33554432 /dev/urandom > /tmp/blob.bin")
    result = b.run(
        ["python", "-c", BENCH],
        env={
            "A_URL": a._base_url,
            "A_HF_TOKEN": a._session.headers["Authorization"].split(" ", 1)[1],
            "A_SBX_TOKEN": a._session.headers["X-Sandbox-Token"],
        },
        timeout=300,
    )
    print(result.stdout)
finally:
    a.kill()
    b.kill()
