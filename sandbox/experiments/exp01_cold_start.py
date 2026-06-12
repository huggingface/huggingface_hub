"""Experiment 1: baseline cold start of a cpu-basic job with an exposed port.

Measures:
- t_api: time for run_job() call to return
- t_running: time until job status == RUNNING
- t_http: time until the exposed URL answers (any HTTP status from the actual server)
- proxy auth behavior (no auth vs Bearer token)
"""

import time

import requests

from huggingface_hub import get_token, inspect_job, run_job


t0 = time.time()
job = run_job(
    image="python:3.12",
    command=["python", "-m", "http.server", "8000"],
    expose=[8000],
    flavor="cpu-basic",
    timeout="10m",
    namespace="Wauplin",
)
t_api = time.time() - t0
print(f"t_api={t_api:.2f}s job_id={job.id}")

while True:
    info = inspect_job(job_id=job.id)
    if info.status.stage == "RUNNING":
        break
    if info.status.stage in ("COMPLETED", "ERROR", "DELETED"):
        raise RuntimeError(f"job ended: {info.status}")
    time.sleep(0.25)
t_running = time.time() - t0
print(f"t_running={t_running:.2f}s expose_urls={info.status.expose_urls}")

url = info.status.expose_urls[0]
token = get_token()

# Poll with token until the real server answers
last_status = None
while True:
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=5)
        if r.status_code != last_status:
            print(f"  +{time.time() - t0:.2f}s status={r.status_code} server={r.headers.get('server')}")
            last_status = r.status_code
        if r.status_code == 200:
            break
    except Exception as e:
        print(f"  +{time.time() - t0:.2f}s exc={type(e).__name__}")
    time.sleep(0.25)
t_http = time.time() - t0
print(f"t_http={t_http:.2f}s")

# Auth behavior
r_noauth = requests.get(url, timeout=10)
print(f"no-auth: status={r_noauth.status_code} body={r_noauth.text[:200]!r}")
r_badtoken = requests.get(url, headers={"Authorization": "Bearer hf_invalid"}, timeout=10)
print(f"bad-token: status={r_badtoken.status_code} body={r_badtoken.text[:200]!r}")

print(f"\nJOB_ID={job.id}")
print(f"URL={url}")
