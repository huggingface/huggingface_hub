"""Experiment 4: end-to-end test of the Sandbox Python client against a real job."""

import time

from huggingface_hub import Sandbox
from huggingface_hub.errors import SandboxCommandError


t0 = time.time()
sbx = Sandbox.create(timeout="15m")
print(f"created + ready in {time.time() - t0:.1f}s -> {sbx!r}")

try:
    # basic run
    r = sbx.run("python --version")
    print("run:", r)

    # argv form
    r = sbx.run(["python", "-c", "print(40 + 2)"])
    assert r.stdout.strip() == "42", r

    # env, cwd, stdin
    r = sbx.run("echo $WHO in $PWD", env={"WHO": "sandbox"}, cwd="/tmp")
    assert r.stdout.strip() == "sandbox in /tmp", r
    r = sbx.run(["wc", "-c"], stdin="hello")
    assert r.stdout.strip() == "5", r

    # error raising
    try:
        sbx.run("python -c 'print(1/0)'")
        raise AssertionError("should have raised")
    except SandboxCommandError as e:
        print("raised as expected:", str(e).splitlines()[0])
        assert e.result.exit_code == 1

    # check=False
    r = sbx.run("exit 7", check=False)
    assert r.exit_code == 7

    # timeout
    r = sbx.run("sleep 60", timeout=2, check=False)
    assert r.timed_out, r
    print("timeout works:", r)

    # streaming callbacks with live arrival
    chunks = []
    t = time.time()
    sbx.run(
        "for i in 1 2 3; do echo tick-$i; sleep 0.5; done",
        on_stdout=lambda d: chunks.append((time.time() - t, d.strip())),
    )
    print("stream timing:", [(f"{ts:.1f}s", d) for ts, d in chunks])
    assert chunks[2][0] - chunks[0][0] > 0.8, "not streamed live!"

    # sequential exec latency
    lat = []
    for _ in range(10):
        t = time.time()
        sbx.run("true")
        lat.append((time.time() - t) * 1000)
    lat.sort()
    print(f"exec('true') latency: median={lat[5]:.0f}ms min={lat[0]:.0f}ms max={lat[-1]:.0f}ms")

    # files
    sbx.files.write("/app/hello.py", "open('/app/out.txt', 'w').write('from sandbox')")
    sbx.run("python /app/hello.py")
    assert sbx.files.read_text("/app/out.txt") == "from sandbox"
    entries = sbx.files.list("/app")
    print("files.list:", [(e.name, e.size) for e in entries])
    assert sbx.files.exists("/app/hello.py") and not sbx.files.exists("/nope")
    sbx.files.delete("/app/out.txt")
    assert not sbx.files.exists("/app/out.txt")

    # larger binary roundtrip + throughput
    blob = bytes(range(256)) * 4096 * 4  # 4 MiB
    t = time.time()
    sbx.files.write("/tmp/blob.bin", blob)
    up = time.time() - t
    t = time.time()
    back = sbx.files.read("/tmp/blob.bin")
    down = time.time() - t
    assert back == blob
    print(f"4MiB upload {4 / up:.1f} MiB/s, download {4 / down:.1f} MiB/s")

    # background process + logs follow + wait
    proc = sbx.spawn("for i in 1 2 3 4 5; do echo bg-$i; sleep 0.2; done", tag="demo")
    print("spawned:", proc, "running:", proc.running)
    lines = [data.strip() for stream, data in proc.logs(follow=True)]
    print("followed:", lines)
    assert proc.wait() == 0
    print("processes:", sbx.processes())

    # long-lived server + url()
    web = sbx.spawn("python -m http.server 8080", tag="web")
    time.sleep(0.8)
    r = sbx.run("python -c \"import urllib.request; print(urllib.request.urlopen('http://localhost:8080').status)\"")
    assert r.stdout.strip() == "200"
    web.kill()
    print("web server spawn/kill ok")

    # reconnection from a 'different machine'
    sbx2 = Sandbox.connect(sbx.id)
    r = sbx2.run("hostname")
    print("reconnected, hostname:", r.stdout.strip())

    # list
    ids = [j.id for j in Sandbox.list()]
    assert sbx.id in ids
    print(f"Sandbox.list: {len(ids)} running, contains ours")

    print("\nALL E2E TESTS PASSED")
finally:
    sbx.kill()
    print("killed")
