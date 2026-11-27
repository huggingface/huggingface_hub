# Copyright 2026-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stdlib-only orchestrator executed inside a lightweight `cpu-basic` Job for HF Jobs sidecars.

IMPORTANT: This module MUST NOT import `huggingface_hub` or any third-party packages. It is
serialized directly into the orchestrator Job (`python:3.12-slim`) and also imported by unit tests.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any


TERMINAL_STAGES = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
MAIN_JOB_ID_PREFIX = "[sidecars] main_job_id="
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _log(message: str) -> None:
    print(f"[sidecars] {message}", flush=True)


class _HubJobsClient:
    """Minimal stdlib HTTP client for the HF Jobs REST API."""

    def __init__(self, endpoint: str, namespace: str, token: str, request_timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.namespace = namespace
        self._token = token
        self.request_timeout = request_timeout

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "huggingface-hub-sidecar-orchestrator",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        max_retries: int = 4,
    ) -> tuple[Any, dict[str, str]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = self._headers(json_body=payload is not None)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    return (json.loads(raw) if raw.strip() else None), resp_headers
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in _RETRYABLE_STATUS_CODES and attempt + 1 < max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {exc.code} on {method} {url}: {body or exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt + 1 < max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
                raise RuntimeError(f"Network error on {method} {url}: {exc}") from exc
        raise RuntimeError(f"Request failed on {method} {url}: {last_error}")

    def create_job(self, job_spec: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}"
        data, _ = self._request_json("POST", url, job_spec)
        return data

    def list_group_jobs(self, group_label_key: str, group_id: str) -> dict[str, dict[str, Any]]:
        """Fetch all Jobs in the sidecar group in O(1) paginated API call(s)."""
        query = urllib.parse.urlencode([("label", f"{group_label_key}={group_id}")])
        url: str | None = f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}?{query}"
        jobs_by_id: dict[str, dict[str, Any]] = {}
        while url:
            data, headers = self._request_json("GET", url)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        jobs_by_id[str(item["id"])] = item
            link = headers.get("link", "")
            match = _LINK_NEXT_RE.search(link)
            url = match.group(1) if match else None
        return jobs_by_id

    def cancel_job(self, job_id: str) -> bool:
        url = f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}/{urllib.parse.quote(job_id, safe='')}/cancel"
        try:
            self._request_json("POST", url)
            return True
        except Exception as exc:
            _log(f"Warning: failed to cancel job {job_id}: {exc}")
            return False

    def fetch_tail_logs(self, job_id: str, tail: int = 40, read_timeout: float = 4.0) -> list[str]:
        """Best-effort fetch of the last `tail` log lines from the SSE `/logs` stream."""
        url = (
            f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}/"
            f"{urllib.parse.quote(job_id, safe='')}/logs?tail={tail}"
        )
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        lines: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=read_timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: {"):
                        continue
                    try:
                        event = json.loads(line[len("data: ") :])
                    except json.JSONDecodeError:
                        continue
                    text = event.get("data")
                    if isinstance(text, str) and not text.startswith("===== Job started"):
                        lines.append(text)
        except Exception:
            # SSE streams stay open for running jobs and end on read timeout once buffered lines flush.
            pass
        return lines[-tail:]


def _probe_health(host: str, healthcheck: dict[str, Any]) -> tuple[bool, str]:
    """Run one health check probe (`http` or `tcp`) against `host`."""
    kind = healthcheck["kind"]
    port = int(healthcheck["port"])
    timeout = float(healthcheck.get("probe_timeout_secs", 5.0))
    if kind == "tcp":
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, f"tcp://{host}:{port} reachable"
        except OSError as exc:
            return False, str(exc)
    if kind == "http":
        path = str(healthcheck.get("path", "/"))
        if not path.startswith("/"):
            path = "/" + path
        url = f"http://{host}:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= status < 300:
                    return True, f"{url} -> {status}"
                return False, f"{url} -> HTTP {status}"
        except urllib.error.HTTPError as exc:
            return False, f"{url} -> HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return False, f"{url} -> {exc}"
    return False, f"Unknown healthcheck kind: {kind}"


def _job_stage(job: dict[str, Any] | None) -> str:
    if not job:
        return "UNKNOWN"
    status = job.get("status")
    if isinstance(status, dict) and isinstance(status.get("stage"), str):
        return status["stage"]
    return "UNKNOWN"


def _job_message(job: dict[str, Any] | None) -> str:
    if not job:
        return ""
    status = job.get("status")
    if isinstance(status, dict) and isinstance(status.get("message"), str):
        return status["message"]
    return ""


def _cancel_jobs_parallel(client: _HubJobsClient, job_ids: list[str]) -> int:
    unique_ids = list(dict.fromkeys(jid for jid in job_ids if jid))
    if not unique_ids:
        return 0
    cancelled = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(unique_ids))) as pool:
        for ok in pool.map(client.cancel_job, unique_ids):
            if ok:
                cancelled += 1
    return cancelled


def _dump_failure_logs(client: _HubJobsClient, name: str, job_id: str) -> None:
    lines = client.fetch_tail_logs(job_id, tail=40)
    if not lines:
        return
    _log(f"--- tail logs for sidecar '{name}' ({job_id}) ---")
    for line in lines:
        print(f"  [{name}] {line}", flush=True)
    _log(f"--- end tail logs for '{name}' ---")


def run_orchestrator(
    manifest: dict[str, Any],
    *,
    token: str,
    child_secrets: dict[str, Any] | None = None,
    network_prefix: str | None = None,
    probe_host_override: str | None = None,
    poll_interval_secs: float = 3.0,
    install_signal_handlers: bool = True,
) -> int:
    """Execute the full sidecar lifecycle and return the orchestrator exit code."""
    endpoint = str(manifest["endpoint"])
    namespace = str(manifest["namespace"])
    group_id = str(manifest["group_id"])
    group_label_key = str(manifest.get("group_label_key", "hf-sidecar-group"))
    sidecars: list[dict[str, Any]] = list(manifest["sidecars"])
    main_spec: dict[str, Any] = dict(manifest["main"]["spec"])
    child_secrets = child_secrets or {}
    sidecar_secrets: dict[str, dict[str, Any]] = dict(child_secrets.get("sidecars") or {})
    main_secrets: dict[str, Any] = dict(child_secrets.get("main") or {})
    if main_secrets:
        main_spec["secrets"] = main_secrets

    prefix = network_prefix if network_prefix is not None else os.environ.get("HF_NETWORK_GROUP_PREFIX", "")
    client = _HubJobsClient(endpoint=endpoint, namespace=namespace, token=token)

    sidecar_jobs: dict[str, str] = {}  # instance_name -> job_id
    sidecar_meta: dict[str, dict[str, Any]] = {sc["name"]: sc for sc in sidecars}
    main_job_id: str | None = None
    terminated_by_signal = threading.Event()
    cleanup_lock = threading.Lock()
    cleaned_up = False

    def cleanup_all(*, exclude_ids: set[str] | None = None) -> int:
        nonlocal cleaned_up
        with cleanup_lock:
            if cleaned_up:
                return 0
            cleaned_up = True
            exclude = exclude_ids or set()
            to_cancel = [jid for jid in sidecar_jobs.values() if jid not in exclude]
            if main_job_id and main_job_id not in exclude:
                to_cancel.append(main_job_id)
            if not to_cancel:
                return 0
            # Filter out jobs already known to be terminal when possible.
            try:
                live = client.list_group_jobs(group_label_key, group_id)
                active_to_cancel = [jid for jid in to_cancel if _job_stage(live.get(jid)) not in TERMINAL_STAGES]
            except Exception:
                active_to_cancel = to_cancel
            return _cancel_jobs_parallel(client, active_to_cancel)

    if install_signal_handlers and threading.current_thread() is threading.main_thread():

        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            _log("Received termination signal; canceling active group jobs...")
            terminated_by_signal.set()
            count = cleanup_all()
            _log(f"Canceled {count} job(s) on signal.")
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        # 1. Submit all sidecars in parallel.
        _log(f"group={group_id} submitting {len(sidecars)} sidecar(s)...")

        def _submit_sidecar(sc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            spec = dict(sc["spec"])
            sc_sec = sidecar_secrets.get(sc["name"])
            if sc_sec:
                spec["secrets"] = sc_sec
            return sc["name"], client.create_job(spec)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(sidecars)))) as pool:
            futures = [pool.submit(_submit_sidecar, sc) for sc in sidecars]
            errors: list[str] = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    sc_name, info = fut.result()
                    jid = str(info["id"])
                    sidecar_jobs[sc_name] = jid
                    _log(f"started sidecar '{sc_name}' id={jid} url={endpoint}/jobs/{namespace}/{jid}")
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                _log(f"ERROR: failed to submit sidecar(s): {'; '.join(errors)}")
                cancelled = cleanup_all()
                _log(f"Canceled {cancelled} sidecar(s) after submission failure.")
                return 1

        # 2. Wait for all sidecars to reach RUNNING and pass health checks.
        submit_time = time.monotonic()
        running_since: dict[str, float] = {}
        healthy: set[str] = set()
        last_probe_at: dict[str, float] = {}
        last_probe_err: dict[str, str] = {}

        while len(healthy) < len(sidecars):
            if terminated_by_signal.is_set():
                return 143

            jobs_by_id = client.list_group_jobs(group_label_key, group_id)
            now = time.monotonic()

            for sc in sidecars:
                name = sc["name"]
                if name in healthy:
                    continue
                jid = sidecar_jobs[name]
                job = jobs_by_id.get(jid)
                stage = _job_stage(job)

                if stage in TERMINAL_STAGES:
                    msg = _job_message(job)
                    detail = f" ({msg})" if msg else ""
                    _log(f"ERROR: sidecar '{name}' ({jid}) exited before becoming ready: stage={stage}{detail}")
                    _dump_failure_logs(client, name, jid)
                    cancelled = cleanup_all(exclude_ids={jid})
                    _log(f"Canceled {cancelled} remaining sidecar(s).")
                    return 1

                if stage != "RUNNING":
                    startup_timeout = float(sc.get("startup_timeout_secs", 300.0))
                    elapsed = now - submit_time
                    if elapsed > startup_timeout:
                        _log(
                            f"ERROR: sidecar '{name}' ({jid}) timed out waiting to start "
                            f"(still '{stage}' after {int(elapsed)}s > startup_timeout={int(startup_timeout)}s)."
                        )
                        cancelled = cleanup_all()
                        _log(f"Canceled {cancelled} sidecar(s).")
                        return 1
                    continue

                # Sidecar is RUNNING.
                if name not in running_since:
                    running_since[name] = now
                    hc = sc.get("healthcheck")
                    if hc:
                        target = f"{hc['kind']}://{prefix}{sc['probe_alias']}:{hc['port']}{hc.get('path', '')}"
                        _log(f"sidecar '{name}' is RUNNING (probing {target})")
                    else:
                        _log(f"sidecar '{name}' is RUNNING (no healthcheck configured; marking ready)")
                        healthy.add(name)
                        continue

                hc = sc.get("healthcheck")
                if not hc:
                    healthy.add(name)
                    continue

                interval = float(hc.get("interval_secs", 2.0))
                if now - last_probe_at.get(name, 0.0) >= interval:
                    last_probe_at[name] = now
                    host = probe_host_override or f"{prefix}{sc['probe_alias']}"
                    ok, detail = _probe_health(host, hc)
                    if ok:
                        _log(f"sidecar '{name}' is healthy ({detail})")
                        healthy.add(name)
                        continue
                    last_probe_err[name] = detail

                readiness_timeout = float(sc.get("readiness_timeout_secs", 600.0))
                ready_elapsed = now - running_since[name]
                if ready_elapsed > readiness_timeout:
                    err_hint = f" (last probe: {last_probe_err[name]})" if name in last_probe_err else ""
                    _log(
                        f"ERROR: sidecar '{name}' ({jid}) failed readiness check after "
                        f"{int(ready_elapsed)}s > readiness_timeout={int(readiness_timeout)}s{err_hint}."
                    )
                    _dump_failure_logs(client, name, jid)
                    cancelled = cleanup_all()
                    _log(f"Canceled {cancelled} sidecar(s).")
                    return 1

            if len(healthy) < len(sidecars):
                time.sleep(min(poll_interval_secs, 1.0))

        # 3. Launch main Job.
        _log("All sidecars ready; starting main job...")
        main_info = client.create_job(main_spec)
        main_job_id = str(main_info["id"])
        main_url = f"{endpoint}/jobs/{namespace}/{main_job_id}"
        print(f"{MAIN_JOB_ID_PREFIX}{main_job_id} url={main_url}", flush=True)

        # 4. Supervise main Job and sidecars until main finishes or a required sidecar dies.
        reported_ignored_dead: set[str] = set()
        while True:
            if terminated_by_signal.is_set():
                return 143

            jobs_by_id = client.list_group_jobs(group_label_key, group_id)
            main_job = jobs_by_id.get(main_job_id)
            main_stage = _job_stage(main_job)

            if main_stage in TERMINAL_STAGES:
                msg = _job_message(main_job)
                detail = f" ({msg})" if msg else ""
                cancelled = cleanup_all(exclude_ids={main_job_id})
                _log(
                    f"Main job {main_job_id} finished with stage={main_stage}{detail}; canceled {cancelled} sidecar(s)."
                )
                return 0 if main_stage == "COMPLETED" else 1

            for name, jid in sidecar_jobs.items():
                sc_job = jobs_by_id.get(jid)
                sc_stage = _job_stage(sc_job)
                if sc_stage not in TERMINAL_STAGES:
                    continue
                on_exit = sidecar_meta[name].get("on_exit", "fail")
                if on_exit == "ignore":
                    if name not in reported_ignored_dead:
                        reported_ignored_dead.add(name)
                        _log(f"Sidecar '{name}' ({jid}) exited with stage={sc_stage} (on_exit=ignore; continuing).")
                    continue
                msg = _job_message(sc_job)
                detail = f" ({msg})" if msg else ""
                _log(
                    f"ERROR: sidecar '{name}' ({jid}) exited unexpectedly with stage={sc_stage}{detail}; aborting run."
                )
                _dump_failure_logs(client, name, jid)
                cancelled = cleanup_all(exclude_ids={jid})
                _log(f"Canceled {cancelled} active job(s) (main + remaining sidecars).")
                return 1

            time.sleep(poll_interval_secs)
    except SystemExit:
        raise
    except Exception as exc:
        _log(f"ERROR: unexpected orchestrator failure: {exc}")
        cancelled = cleanup_all()
        _log(f"Canceled {cancelled} active job(s).")
        return 1


def _main() -> None:
    raw_manifest = os.environ.get("HF_ORCHESTRATOR_MANIFEST")
    token = os.environ.get("HF_ORCHESTRATOR_TOKEN")
    if not raw_manifest or not token:
        _log("ERROR: missing HF_ORCHESTRATOR_MANIFEST or HF_ORCHESTRATOR_TOKEN.")
        sys.exit(2)
    manifest = json.loads(zlib.decompress(base64.b64decode(raw_manifest)).decode("utf-8"))
    raw_secrets = os.environ.get("HF_ORCHESTRATOR_SECRETS")
    child_secrets = json.loads(raw_secrets) if raw_secrets else {}
    poll_interval = float(manifest.get("poll_interval_secs", 3.0))
    exit_code = run_orchestrator(manifest, token=token, child_secrets=child_secrets, poll_interval_secs=poll_interval)
    sys.exit(exit_code)


if __name__ == "__main__":
    _main()
