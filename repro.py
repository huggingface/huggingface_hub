"""
Reproduces OSError when 8 processes call from_pretrained concurrently on a cold cache.
pip install transformers==5.3.0 huggingface_hub==1.8.0 hf-xet==1.4.3 torch
Typically fails within 5-30 rounds (~10% per-round failure rate).
"""

import os
import shutil
import sys
from multiprocessing import Process, Queue


def worker(rank, model_id, cache_dir, q):
    os.environ["HF_HOME"] = cache_dir
    try:
        from transformers import AutoModelForCausalLM

        AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=f"{cache_dir}/hub", device_map="meta"
        )
        q.put(("ok", ""))
    except Exception as e:
        import traceback

        q.put(("fail", traceback.format_exc()))


model = "Qwen/Qwen3.5-4B"
for r in range(1, 51):
    cache = "/tmp/repro_hf_cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache)
    q = Queue()
    procs = [Process(target=worker, args=(i, model, cache, q)) for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    results = [q.get_nowait() for _ in range(q.qsize())]
    fails = [tb for s, tb in results if s == "fail"]
    if fails:
        print(f"FAILED round {r}:\n{fails[0]}")
        sys.exit(1)
    else:
        print(f"round {r}: 8/8 ok")
