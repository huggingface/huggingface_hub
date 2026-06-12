"""Experiment 2: what does the hf.jobs proxy support?

- chunked streaming (does output arrive incrementally or buffered?)
- SSE
- WebSocket upgrade
- multiple exposed ports
"""

import time

import requests

from huggingface_hub import get_token, inspect_job, run_job


APP = """
import asyncio, datetime
from aiohttp import web

async def chunk(request):
    resp = web.StreamResponse()
    resp.headers["Content-Type"] = "text/plain"
    await resp.prepare(request)
    for i in range(5):
        await resp.write(f"chunk-{i} {datetime.datetime.now().isoformat()}\\n".encode())
        await asyncio.sleep(0.5)
    await resp.write_eof()
    return resp

async def sse(request):
    resp = web.StreamResponse()
    resp.headers["Content-Type"] = "text/event-stream"
    await resp.prepare(request)
    for i in range(5):
        await resp.write(f"data: event-{i}\\n\\n".encode())
        await asyncio.sleep(0.5)
    await resp.write_eof()
    return resp

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        await ws.send_str(f"echo:{msg.data}")
        if msg.data == "close":
            await ws.close()
    return ws

async def ok(request):
    return web.Response(text="ok")

app = web.Application()
app.router.add_get("/", ok)
app.router.add_get("/chunk", chunk)
app.router.add_get("/sse", sse)
app.router.add_get("/ws", ws_handler)
web.run_app(app, port=8000)
"""

job = run_job(
    image="python:3.12",
    command=[
        "bash",
        "-c",
        "pip install -q aiohttp && (python -m http.server 8001 &) && python -c \"import os; exec(os.environ['APP'])\"",
    ],
    env={"APP": APP},
    expose=[8000, 8001],
    flavor="cpu-basic",
    timeout="15m",
    namespace="Wauplin",
)
print(f"job_id={job.id}")

t0 = time.time()
while True:
    info = inspect_job(job_id=job.id)
    if info.status.stage == "RUNNING":
        break
    if info.status.stage in ("COMPLETED", "ERROR", "DELETED"):
        raise RuntimeError(f"job ended: {info.status}")
    time.sleep(0.5)
print(f"running at +{time.time() - t0:.1f}s urls={info.status.expose_urls}")

urls = {u.split("--")[1].split(".")[0]: u for u in info.status.expose_urls}
token = get_token()
s = requests.Session()
s.headers["Authorization"] = f"Bearer {token}"

# wait for app ready (pip install takes a while)
while True:
    try:
        if s.get(urls["8000"], timeout=5).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)
print(f"app ready at +{time.time() - t0:.1f}s")

# 1. chunked streaming: measure inter-chunk arrival times
print("\n--- chunked streaming ---")
t = time.time()
r = s.get(urls["8000"] + "/chunk", stream=True, timeout=30)
for line in r.iter_lines():
    print(f"  +{time.time() - t:.2f}s {line.decode()[:40]}")

# 2. SSE
print("\n--- SSE ---")
t = time.time()
r = s.get(urls["8000"] + "/sse", stream=True, timeout=30)
for line in r.iter_lines():
    if line:
        print(f"  +{time.time() - t:.2f}s {line.decode()[:40]}")

# 3. multi-port
print("\n--- multi-port (8001 = http.server) ---")
r = s.get(urls["8001"], timeout=10)
print(f"  status={r.status_code} server={r.headers.get('server')}")

# 4. websocket
print("\n--- websocket ---")
try:
    import websockets.sync.client as wsc

    ws_url = urls["8000"].replace("https://", "wss://") + "/ws"
    with wsc.connect(ws_url, additional_headers={"Authorization": f"Bearer {token}"}) as ws:
        t = time.time()
        for i in range(5):
            ws.send(f"ping-{i}")
            reply = ws.recv()
        rtt = (time.time() - t) / 5 * 1000
        print(f"  websocket OK, reply={reply}, avg rtt={rtt:.0f}ms")
except Exception as e:
    print(f"  websocket FAILED: {type(e).__name__}: {e}")

print(f"\nJOB_ID={job.id}")
