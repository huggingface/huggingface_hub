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
import email.utils
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
from datetime import datetime, timezone
from typing import Any


TERMINAL_STAGES = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})
MAIN_JOB_ID_PREFIX = "[sidecars] main_job_id="
_LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_SECS = 60.0
_SIGTERM_HTTP_TIMEOUT_SECS = 1.5


class _TransientHubError(RuntimeError):
    """Retryable Hub API or network error (5xx, 429, 408, timeouts, connection errors)."""


class _PermanentHubError(RuntimeError):
    """Non-retryable Hub API error (e.g. 400, 401, 403, 404, 422)."""


def _log(message: str) -> None:
    print(f"[sidecars] {message}", flush=True)


def _parse_retry_after(headers: Any) -> float | None:
    """Parse an HTTP `Retry-After` header (seconds or HTTP-date) into a capped delay."""
    if headers is None:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    raw_str = str(raw).strip()
    try:
        delay = float(raw_str)
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(raw_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delay = (dt - datetime.now(timezone.utc)).total_seconds()
        except Exception:
            return None
    if delay <= 0:
        return 0.0
    return min(delay, _MAX_RETRY_AFTER_SECS)


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
        retry_on_network_or_5xx: bool = True,
        retry_on_429: bool = True,
        timeout: float | None = None,
    ) -> tuple[Any, dict[str, str]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = self._headers(json_body=payload is not None)
        req_timeout = self.request_timeout if timeout is None else timeout
        last_error: Exception | None = None
        for attempt in range(max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    return (json.loads(raw) if raw.strip() else None), resp_headers
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429 and retry_on_429 and attempt + 1 < max_retries:
                    delay = _parse_retry_after(exc.headers)
                    time.sleep(min(2.0**attempt, 8.0) if delay is None else delay)
                    continue
                if (
                    exc.code != 429
                    and exc.code in _RETRYABLE_STATUS_CODES
                    and retry_on_network_or_5xx
                    and attempt + 1 < max_retries
                ):
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                err_cls = _TransientHubError if exc.code in _RETRYABLE_STATUS_CODES else _PermanentHubError
                raise err_cls(f"HTTP {exc.code} on {method} {url}: {body or exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if retry_on_network_or_5xx and attempt + 1 < max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
                raise _TransientHubError(f"Network error on {method} {url}: {exc}") from exc
        raise _TransientHubError(f"Request failed on {method} {url}: {last_error}")

    def create_job(self, job_spec: dict[str, Any]) -> dict[str, Any]:
        # POST /api/jobs is not idempotent: retry only 429 (rejected before creation), never timeout/5xx.
        url = f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}"
        data, _ = self._request_json("POST", url, job_spec, retry_on_network_or_5xx=False, retry_on_429=True)
        return data

    def list_group_jobs(
        self,
        group_label_key: str,
        group_id: str,
        *,
        timeout: float | None = None,
        max_retries: int = 4,
    ) -> dict[str, dict[str, Any]]:
        """Fetch all Jobs in the sidecar group in O(1) paginated API call(s)."""
        query = urllib.parse.urlencode([("label", f"{group_label_key}={group_id}")])
        url: str | None = f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}?{query}"
        jobs_by_id: dict[str, dict[str, Any]] = {}
        while url:
            data, headers = self._request_json("GET", url, timeout=timeout, max_retries=max_retries)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        jobs_by_id[str(item["id"])] = item
            link = headers.get("link", "")
            match = _LINK_NEXT_RE.search(link)
            url = match.group(1) if match else None
        return jobs_by_id

    def cancel_job(self, job_id: str, *, timeout: float | None = None, max_retries: int = 4) -> bool:
        url = f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}/{urllib.parse.quote(job_id, safe='')}/cancel"
        try:
            self._request_json("POST", url, timeout=timeout, max_retries=max_retries)
            return True
        except Exception as exc:
            _log(f"Warning: failed to cancel job {job_id}: {exc}")
            return False

    def fetch_tail_logs(
        self,
        job_id: str,
        tail: int = 40,
        read_timeout: float = 4.0,
        total_timeout: float = 5.0,
    ) -> list[str]:
        """Best-effort fetch of the last `tail` log lines from the SSE `/logs` stream."""
        url = (
            f"{self.endpoint}/api/jobs/{urllib.parse.quote(self.namespace, safe='')}/"
            f"{urllib.parse.quote(job_id, safe='')}/logs?tail={tail}"
        )
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        lines: list[str] = []
        deadline = time.monotonic() + total_timeout
        try:
            with urllib.request.urlopen(req, timeout=min(read_timeout, total_timeout)) as resp:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    raw_sock = getattr(getattr(resp, "fp", None), "raw", None)
                    sock = getattr(raw_sock, "_sock", None)
                    if sock is not None:
                        try:
                            sock.settimeout(min(read_timeout, remaining))
                        except Exception:
                            pass
                    raw_line = resp.readline()
                    if not raw_line:
                        break
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
            # SSE streams stay open for running jobs and end on read timeout or wall-clock deadline.
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


def _cancel_jobs_parallel(
    client: _HubJobsClient,
    job_ids: list[str],
    *,
    timeout: float | None = None,
    max_retries: int = 4,
) -> int:
    unique_ids = list(dict.fromkeys(jid for jid in job_ids if jid))
    if not unique_ids:
        return 0
    cancelled = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(unique_ids))) as pool:
        futures = [pool.submit(client.cancel_job, jid, timeout=timeout, max_retries=max_retries) for jid in unique_ids]
        for fut in futures:
            if fut.result():
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
    heartbeat_interval_secs: float = 25.0,
    install_signal_handlers: bool = True,
) -> int:
    """Execute the full sidecar lifecycle and return the orchestrator exit code."""
    endpoint = str(manifest["endpoint"])
    namespace = str(manifest["namespace"])
    group_id = str(manifest["group_id"])
    group_label_key = str(manifest.get("group_label_key", "hf-sidecar-group"))
    role_label_key = str(manifest.get("role_label_key", "hf-sidecar-role"))
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
    sidecar_jobs_lock = threading.Lock()
    sidecar_meta: dict[str, dict[str, Any]] = {sc["name"]: sc for sc in sidecars}
    main_job_id: str | None = None
    terminated_by_signal = threading.Event()
    cleanup_lock = threading.Lock()
    cleaned_up = False

    def _is_orchestrator_job(job: dict[str, Any]) -> bool:
        labels = job.get("labels")
        return isinstance(labels, dict) and labels.get(role_label_key) == "orchestrator"

    def _snapshot_tracked_ids(exclude: set[str]) -> list[str]:
        with sidecar_jobs_lock:
            ids = [jid for jid in sidecar_jobs.values() if jid not in exclude]
        if main_job_id and main_job_id not in exclude:
            ids.append(main_job_id)
        return list(dict.fromkeys(ids))

    def cleanup_all(*, exclude_ids: set[str] | None = None, fast: bool = False) -> int:
        nonlocal cleaned_up
        with cleanup_lock:
            if cleaned_up:
                return 0
            cleaned_up = True
            exclude = exclude_ids or set()
            tracked = _snapshot_tracked_ids(exclude)

            if fast:
                # On SIGTERM the platform grace period is ~2s: fire tracked cancels immediately
                # without waiting behind a list round-trip, while sweeping the group label concurrently
                # to catch any untracked/in-flight jobs.
                cancelled = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(2, len(tracked) + 1))) as pool:
                    list_future = pool.submit(
                        client.list_group_jobs,
                        group_label_key,
                        group_id,
                        timeout=_SIGTERM_HTTP_TIMEOUT_SECS,
                        max_retries=1,
                    )
                    cancel_futures = [
                        pool.submit(client.cancel_job, jid, timeout=_SIGTERM_HTTP_TIMEOUT_SECS, max_retries=1)
                        for jid in tracked
                    ]
                    for fut in cancel_futures:
                        if fut.result():
                            cancelled += 1
                    try:
                        live = list_future.result()
                    except Exception:
                        live = {}
                    tracked_set = set(tracked)
                    extra_ids = [
                        jid
                        for jid, job in live.items()
                        if jid not in exclude
                        and jid not in tracked_set
                        and not _is_orchestrator_job(job)
                        and _job_stage(job) not in TERMINAL_STAGES
                    ]
                    if extra_ids:
                        cancelled += _cancel_jobs_parallel(
                            client, extra_ids, timeout=_SIGTERM_HTTP_TIMEOUT_SECS, max_retries=1
                        )
                return cancelled

            # Normal teardown: cancel all non-terminal non-orchestrator jobs in the group label
            # (catches any untracked/ghost jobs) plus any tracked jobs not yet visible in list.
            try:
                live = client.list_group_jobs(group_label_key, group_id)
                active_to_cancel = [
                    jid
                    for jid, job in live.items()
                    if jid not in exclude and not _is_orchestrator_job(job) and _job_stage(job) not in TERMINAL_STAGES
                ]
                for jid in tracked:
                    if jid not in live:
                        active_to_cancel.append(jid)
            except Exception:
                active_to_cancel = tracked
            return _cancel_jobs_parallel(client, active_to_cancel)

    if install_signal_handlers and threading.current_thread() is threading.main_thread():

        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            _log("Received termination signal; canceling active group jobs...")
            terminated_by_signal.set()
            count = cleanup_all(fast=True)
            _log(f"Canceled {count} job(s) on signal.")
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        # 1. Submit all sidecars in parallel.
        _log(f"group={group_id} submitting {len(sidecars)} sidecar(s)...")

        def _submit_sidecar(sc: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
            if terminated_by_signal.is_set():
                return sc["name"], None
            spec = dict(sc["spec"])
            sc_sec = sidecar_secrets.get(sc["name"])
            if sc_sec:
                spec["secrets"] = sc_sec
            info = client.create_job(spec)
            jid = str(info["id"])
            with sidecar_jobs_lock:
                sidecar_jobs[sc["name"]] = jid
            # If SIGTERM arrived while this POST was in flight, cancel immediately in case cleanup_all already ran.
            if terminated_by_signal.is_set():
                client.cancel_job(jid, timeout=_SIGTERM_HTTP_TIMEOUT_SECS, max_retries=1)
            return sc["name"], info

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(sidecars)))) as pool:
            futures = [pool.submit(_submit_sidecar, sc) for sc in sidecars]
            errors: list[str] = []
            logged_expose_urls: set[str] = set()
            for fut in concurrent.futures.as_completed(futures):
                try:
                    sc_name, info = fut.result()
                    if info is None:
                        continue
                    jid = str(info["id"])
                    expose_urls = (info.get("status") or {}).get("exposeUrls") or []
                    if expose_urls:
                        logged_expose_urls.add(sc_name)
                    expose_suffix = f" expose={','.join(expose_urls)}" if expose_urls else ""
                    _log(f"started sidecar '{sc_name}' id={jid} url={endpoint}/jobs/{namespace}/{jid}{expose_suffix}")
                except Exception as exc:
                    errors.append(str(exc))
            if terminated_by_signal.is_set():
                return 143
            if errors:
                _log(f"ERROR: failed to submit sidecar(s): {'; '.join(errors)}")
                cancelled = cleanup_all()
                _log(f"Canceled {cancelled} sidecar(s) after submission failure.")
                return 1

        # 2. Wait for all sidecars to reach RUNNING and pass health checks.
        submit_time = time.monotonic()
        last_heartbeat = submit_time
        running_since: dict[str, float] = {}
        healthy: set[str] = set()
        reported_ignored_dead: set[str] = set()
        last_probe_at: dict[str, float] = {}
        last_probe_err: dict[str, str] = {}
        consecutive_poll_errors = 0

        while len(healthy) < len(sidecars):
            if terminated_by_signal.is_set():
                return 143

            try:
                jobs_by_id = client.list_group_jobs(group_label_key, group_id)
                if consecutive_poll_errors:
                    _log(f"Hub API polling recovered after {consecutive_poll_errors} transient failure(s).")
                    consecutive_poll_errors = 0
            except _TransientHubError as exc:
                consecutive_poll_errors += 1
                backoff = min(poll_interval_secs * (2 ** min(consecutive_poll_errors - 1, 4)), 60.0)
                if consecutive_poll_errors == 1 or consecutive_poll_errors % 5 == 0:
                    _log(
                        f"Warning: transient Hub API error while waiting for sidecars "
                        f"(attempt {consecutive_poll_errors}, retrying in {backoff:.1f}s): {exc}"
                    )
                time.sleep(backoff)
                continue

            now = time.monotonic()
            due_probes: list[tuple[str, str, dict[str, Any]]] = []

            for sc in sidecars:
                name = sc["name"]
                jid = sidecar_jobs[name]
                job = jobs_by_id.get(jid)
                stage = _job_stage(job)

                if stage in TERMINAL_STAGES:
                    on_exit = sc.get("on_exit", "fail")
                    fast_ignored_complete = (
                        on_exit == "ignore" and stage == "COMPLETED" and sc.get("healthcheck") is None
                    )
                    if on_exit == "ignore" and (name in healthy or fast_ignored_complete):
                        healthy.add(name)
                        if name not in reported_ignored_dead:
                            reported_ignored_dead.add(name)
                            _log(f"Sidecar '{name}' ({jid}) exited with stage={stage} (on_exit=ignore; continuing).")
                        continue
                    msg = _job_message(job)
                    detail = f" ({msg})" if msg else ""
                    phase = "while waiting for other sidecars" if name in healthy else "before becoming ready"
                    _log(f"ERROR: sidecar '{name}' ({jid}) exited {phase}: stage={stage}{detail}")
                    cancelled = cleanup_all(exclude_ids={jid})
                    _log(f"Canceled {cancelled} remaining sidecar(s).")
                    _dump_failure_logs(client, name, jid)
                    return 1

                if name in healthy:
                    continue

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
                    expose_urls = ((job or {}).get("status") or {}).get("exposeUrls") or []
                    if expose_urls and name not in logged_expose_urls:
                        logged_expose_urls.add(name)
                        _log(f"sidecar '{name}' exposed at {', '.join(expose_urls)}")
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
                    due_probes.append((name, host, hc))

            # Run due health probes concurrently so many replicas don't serialize probe timeouts.
            if due_probes:
                if len(due_probes) == 1:
                    p_name, p_host, p_hc = due_probes[0]
                    probe_results = [(p_name, _probe_health(p_host, p_hc))]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(due_probes))) as pool:
                        probe_futs = [
                            (p_name, pool.submit(_probe_health, p_host, p_hc)) for p_name, p_host, p_hc in due_probes
                        ]
                        probe_results = [(p_name, fut.result()) for p_name, fut in probe_futs]
                for p_name, (ok, detail) in probe_results:
                    if ok:
                        _log(f"sidecar '{p_name}' is healthy ({detail})")
                        healthy.add(p_name)
                    else:
                        last_probe_err[p_name] = detail

            # Check readiness timeouts after probes complete.
            now = time.monotonic()
            for sc in sidecars:
                name = sc["name"]
                if name in healthy or name not in running_since:
                    continue
                readiness_timeout = float(sc.get("readiness_timeout_secs", 600.0))
                ready_elapsed = now - running_since[name]
                if ready_elapsed > readiness_timeout:
                    jid = sidecar_jobs[name]
                    err_hint = f" (last probe: {last_probe_err[name]})" if name in last_probe_err else ""
                    _log(
                        f"ERROR: sidecar '{name}' ({jid}) failed readiness check after "
                        f"{int(ready_elapsed)}s > readiness_timeout={int(readiness_timeout)}s{err_hint}."
                    )
                    # Cancel first so billing stops immediately and the SSE log stream closes.
                    cancelled = cleanup_all()
                    _log(f"Canceled {cancelled} sidecar(s).")
                    _dump_failure_logs(client, name, jid)
                    return 1

            if len(healthy) < len(sidecars):
                if heartbeat_interval_secs > 0 and now - last_heartbeat >= heartbeat_interval_secs:
                    last_heartbeat = now
                    parts: list[str] = []
                    for sc in sidecars:
                        sc_name = sc["name"]
                        if sc_name in healthy:
                            parts.append(f"{sc_name}=healthy")
                        else:
                            st = _job_stage(jobs_by_id.get(sidecar_jobs[sc_name]))
                            err = last_probe_err.get(sc_name)
                            parts.append(f"{sc_name}={st} ({err})" if err and st == "RUNNING" else f"{sc_name}={st}")
                    _log(f"waiting ({int(now - submit_time)}s): {', '.join(parts)}")
                time.sleep(min(poll_interval_secs, 1.0))

        # 3. Launch main Job.
        _log("All sidecars ready; starting main job...")
        main_info = client.create_job(main_spec)
        main_job_id = str(main_info["id"])
        if terminated_by_signal.is_set():
            client.cancel_job(main_job_id, timeout=_SIGTERM_HTTP_TIMEOUT_SECS, max_retries=1)
            return 143
        main_url = f"{endpoint}/jobs/{namespace}/{main_job_id}"
        print(f"{MAIN_JOB_ID_PREFIX}{main_job_id} url={main_url}", flush=True)

        # 4. Supervise main Job and sidecars until main finishes or a required sidecar dies.
        consecutive_poll_errors = 0
        while True:
            if terminated_by_signal.is_set():
                return 143

            try:
                jobs_by_id = client.list_group_jobs(group_label_key, group_id)
                if consecutive_poll_errors:
                    _log(f"Hub API polling recovered after {consecutive_poll_errors} transient failure(s).")
                    consecutive_poll_errors = 0
            except _TransientHubError as exc:
                consecutive_poll_errors += 1
                backoff = min(poll_interval_secs * (2 ** min(consecutive_poll_errors - 1, 4)), 60.0)
                if consecutive_poll_errors == 1 or consecutive_poll_errors % 5 == 0:
                    _log(
                        f"Warning: transient Hub API error while supervising main job "
                        f"(attempt {consecutive_poll_errors}, retrying in {backoff:.1f}s): {exc}"
                    )
                time.sleep(backoff)
                continue

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
                cancelled = cleanup_all(exclude_ids={jid})
                _log(f"Canceled {cancelled} active job(s) (main + remaining sidecars).")
                _dump_failure_logs(client, name, jid)
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
