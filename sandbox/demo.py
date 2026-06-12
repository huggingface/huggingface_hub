"""hf sandbox — end-to-end demo.

A realistic agentic workflow: spin up a sandbox, generate + run code, install
dependencies, serve a web app through the public proxy, move artifacts, fan out
parallel sandboxes, reconnect by id. Run with: .venv/bin/python sandbox/demo.py
"""

import concurrent.futures
import time

import requests

from huggingface_hub import Sandbox, get_token
from huggingface_hub.errors import SandboxCommandError


def step(title: str) -> None:
    print(f"\n\033[1m── {title} ──\033[0m")


total_t0 = time.time()

step("1. Create a sandbox (python:3.12, cpu-basic)")
t0 = time.time()
sbx = Sandbox.create(timeout="15m", expose=[7860])
print(f"   ready in {time.time() - t0:.1f}s -> {sbx!r}")

step("2. Run commands (~100ms each)")
print("  ", sbx.run("python --version").stdout.strip())
print("  ", sbx.run(["uname", "-m"]).stdout.strip())

step("3. Write code, run it, read results")
sbx.files.write(
    "/app/fib.py",
    "import json\n"
    "fib = [0, 1]\n"
    "[fib.append(fib[-1] + fib[-2]) for _ in range(20)]\n"
    "json.dump(fib, open('/app/fib.json', 'w'))\n"
    "print(f'computed {len(fib)} numbers')\n",
)
print("  ", sbx.run("python /app/fib.py").stdout.strip())
print("   /app/fib.json:", sbx.files.read_text("/app/fib.json")[:60], "...")

step("4. Errors surface nicely")
try:
    sbx.run("python -c 'import nonexistent_pkg'")
except SandboxCommandError as e:
    print("   SandboxCommandError:", str(e).splitlines()[-1].strip())

step("5. Install a package, stream the output live")
sbx.run("pip install -q flask", on_stdout=lambda d: print("   " + d.strip()) if d.strip() else None)
print("  ", sbx.run("python -c 'import flask; print(f\"flask {flask.__version__}\")'").stdout.strip())

step("6. Spawn a web server, reach it through the public proxy URL")
sbx.files.write(
    "/app/web.py",
    "from flask import Flask\n"
    "app = Flask(__name__)\n"
    "@app.get('/')\n"
    "def home(): return {'hello': 'from a sandbox'}\n"
    "app.run(host='0.0.0.0', port=7860)\n",
)
web = sbx.spawn("python /app/web.py", tag="web")
time.sleep(1.5)
url = sbx.url(7860)
response = requests.get(url, headers={"Authorization": f"Bearer {get_token()}"}, timeout=10)
print(f"   GET {url}")
print(f"   -> {response.status_code} {response.json()}")
web.kill()

step("7. Fan out: 3 sandboxes in parallel, one task each")
t0 = time.time()


def worker(n: int) -> str:
    with Sandbox.create(timeout="10m") as worker_sbx:
        return worker_sbx.run(f"python -c 'print(sum(i*i for i in range({n}_000_000)))'").stdout.strip()


with concurrent.futures.ThreadPoolExecutor(3) as pool:
    sums = list(pool.map(worker, [1, 2, 3]))
print(f"   3 results in {time.time() - t0:.1f}s wall: {[s[:12] + '...' for s in sums]}")

step("8. Reconnect by id (works from any machine, no local state)")
again = Sandbox.connect(sbx.id)
print("  ", again.run("echo still here: $(hostname)").stdout.strip())

step("9. Clean up")
sbx.kill()
print(f"   killed. Total demo time: {time.time() - total_t0:.1f}s")
