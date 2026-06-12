"""Experiment 3: binary distribution into arbitrary images.

Strategy A: mount HF model repo (private) as volume, exec binary directly.
Tests: exec bit preserved? private repo mountable? cold-start overhead of volume?
Image: alpine:3.20 (no Python at all) — proves the 'any image' claim.
"""

import secrets
import sys
import time

import requests

from huggingface_hub import Volume, get_token, inspect_job, run_job


def wait_ready(job, sbx_token, t0):
    info = None
    while True:
        info = inspect_job(job_id=job.id)
        if info.status.stage == "RUNNING":
            break
        if info.status.stage in ("COMPLETED", "ERROR", "DELETED"):
            from huggingface_hub import fetch_job_logs

            print(f"  job ended: {info.status.stage} {info.status.message}")
            for log in fetch_job_logs(job_id=job.id):
                print("  log:", log)
            return None
    t_running = time.time() - t0
    url = info.status.expose_urls[0]
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {get_token()}"
    s.headers["X-Sandbox-Token"] = sbx_token
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            r = s.get(url + "/health", timeout=3)
            if r.status_code == 200:
                t_ready = time.time() - t0
                print(f"  RUNNING at +{t_running:.1f}s, /health OK at +{t_ready:.1f}s -> {r.json()}")
                # verify exec works
                r = s.post(url + "/v1/exec", json={"cmd": "uname -a && id && cat /etc/os-release | head -1"}, stream=True)
                for line in r.iter_lines():
                    print("   ", line.decode())
                return t_ready
        except Exception:
            pass
        time.sleep(0.2)
    print("  TIMED OUT waiting for /health")
    return None


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}

    if which in ("a", "all"):
        print("=== A: alpine:3.20 + mounted volume, direct exec ===")
        sbx_token = secrets.token_urlsafe(32)
        t0 = time.time()
        job = run_job(
            image="alpine:3.20",
            command=["/sbx/sbx-server"],
            volumes=[Volume(type="model", source="Wauplin/sbx-server", mount_path="/sbx", read_only=True)],
            secrets={"SBX_TOKEN": sbx_token},
            expose=[8000],
            flavor="cpu-basic",
            timeout="10m",
            namespace="Wauplin",
        )
        print(f"  job {job.id}")
        results["A_mount_direct"] = wait_ready(job, sbx_token, t0)

    if which in ("b", "all"):
        print("=== B: alpine:3.20 + mounted volume, sh cp+chmod fallback ===")
        sbx_token = secrets.token_urlsafe(32)
        t0 = time.time()
        job = run_job(
            image="alpine:3.20",
            command=["/bin/sh", "-c", "cp /sbx/sbx-server /tmp/.sbx && chmod +x /tmp/.sbx && exec /tmp/.sbx"],
            volumes=[Volume(type="model", source="Wauplin/sbx-server", mount_path="/sbx", read_only=True)],
            secrets={"SBX_TOKEN": sbx_token},
            expose=[8000],
            flavor="cpu-basic",
            timeout="10m",
            namespace="Wauplin",
        )
        print(f"  job {job.id}")
        results["B_mount_cp"] = wait_ready(job, sbx_token, t0)

    if which in ("c", "all"):
        print("=== C: alpine:3.20 + wget download from hf.co (public repo needed? testing private+token) ===")
        sbx_token = secrets.token_urlsafe(32)
        t0 = time.time()
        job = run_job(
            image="alpine:3.20",
            command=[
                "/bin/sh",
                "-c",
                'wget -q --header="Authorization: Bearer $DL_TOKEN" -O /tmp/.sbx '
                "https://huggingface.co/Wauplin/sbx-server/resolve/main/sbx-server "
                "&& chmod +x /tmp/.sbx && exec /tmp/.sbx",
            ],
            secrets={"SBX_TOKEN": sbx_token, "DL_TOKEN": get_token()},
            expose=[8000],
            flavor="cpu-basic",
            timeout="10m",
            namespace="Wauplin",
        )
        print(f"  job {job.id}")
        results["C_download"] = wait_ready(job, sbx_token, t0)

    print("\n=== results ===")
    for k, v in results.items():
        print(f"  {k}: {f'{v:.1f}s' if v else 'FAILED'}")


if __name__ == "__main__":
    main()
