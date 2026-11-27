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
"""Unit tests for Jobs sidecars parsing, orchestrator state machine, HfApi, and CLI."""

from __future__ import annotations

import base64
import json
import socket
import threading
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from huggingface_hub import HfApi, JobOwner, JobStage, Volume
from huggingface_hub._jobs_sidecar_orchestrator import run_orchestrator
from huggingface_hub._jobs_sidecars import (
    SIDECAR_GROUP_LABEL,
    SIDECAR_NAME_LABEL,
    SIDECAR_ROLE_LABEL,
    build_orchestrator_job_spec,
    format_sidecars_summary,
    load_sidecars,
    parse_main_job_marker,
)
from huggingface_hub.cli.hf import app
from huggingface_hub.errors import CLIError


# ---------------------------------------------------------------------------
# Parser & validation tests
# ---------------------------------------------------------------------------


def test_load_sidecars_yaml_full(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hf-jobs.yaml"
    cfg_path.write_text(
        """
defaults:
  startup_timeout: 3m
  readiness_timeout: 7m
  on_exit: fail

sidecars:
  vllm:
    image: vllm/vllm-openai:latest
    command: ["vllm", "serve", "Qwen/Qwen2.5-3B-Instruct", "--port", "8000"]
    flavor: a10g-small
    env:
      VLLM_LOGGING_LEVEL: INFO
    secrets:
      - HF_TOKEN
    volumes:
      - hf://buckets/me/cache:/cache
    healthcheck:
      http: "8000/health"
      interval: 1s
      timeout: 3s
  worker:
    image: python:3.12-slim
    command: "python -m http.server 8786"
    flavor: cpu-upgrade
    replicas: 3
    healthcheck:
      tcp: 8786
    on_exit: ignore
""",
        encoding="utf-8",
    )

    config = load_sidecars(cfg_path, environ={"HF_TOKEN": "hf_secret_123"})
    assert set(config.sidecars) == {"vllm", "worker"}
    assert config.total_instances == 4

    vllm = config.sidecars["vllm"]
    assert vllm.flavor == "a10g-small"
    assert vllm.startup_timeout_secs == 180
    assert vllm.readiness_timeout_secs == 420
    assert vllm.secrets == {"HF_TOKEN": "hf_secret_123"}
    assert vllm.volumes == [Volume(type="bucket", source="me/cache", mount_path="/cache")]
    assert vllm.healthcheck is not None
    assert vllm.healthcheck.kind == "http"
    assert vllm.healthcheck.port == 8000
    assert vllm.healthcheck.path == "/health"
    assert vllm.healthcheck.interval_secs == 1.0
    assert vllm.healthcheck.probe_timeout_secs == 3.0

    worker = config.sidecars["worker"]
    assert worker.command == ["sh", "-c", "python -m http.server 8786"]
    assert worker.replicas == 3
    assert worker.on_exit == "ignore"
    assert worker.healthcheck is not None
    assert worker.healthcheck.kind == "tcp"
    assert worker.healthcheck.port == 8786

    instances = config.expand_instances()
    assert [i.instance_name for i in instances] == ["vllm", "worker-0", "worker-1", "worker-2"]
    assert instances[0].network_aliases == ["vllm"]
    assert instances[1].network_aliases == ["worker", "worker-0"]
    assert instances[1].probe_alias == "worker-0"

    summary = format_sidecars_summary(config)
    assert "vllm (flavor=a10g-small, health=http:8000/health, secrets=[HF_TOKEN=***])" in summary
    assert "worker (flavor=cpu-upgrade, replicas=3, health=tcp:8786, on_exit=ignore)" in summary
    assert "hf_secret_123" not in summary


def test_load_sidecars_python_shorthand_dict() -> None:
    config = load_sidecars(
        {
            "vllm-1": {
                "image": "vllm/vllm-openai:latest",
                "command": ["vllm", "serve", "model"],
                "secrets": {"HF_TOKEN": "hf_inline_token"},
                "healthcheck": {"http": {"port": 8000, "path": "/health"}},
            },
            "vllm-2": {
                "image": "vllm/vllm-openai:latest",
                "command": ["vllm", "serve", "model"],
            },
        }
    )
    assert list(config.sidecars) == ["vllm-1", "vllm-2"]
    assert config.sidecars["vllm-1"].secrets == {"HF_TOKEN": "hf_inline_token"}
    assert config.sidecars["vllm-1"].healthcheck is not None
    assert config.sidecars["vllm-1"].healthcheck.path == "/health"


def test_load_sidecars_yaml_rejects_inline_secret_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hf-jobs.yaml"
    cfg_path.write_text(
        """
sidecars:
  srv:
    image: python:3.12
    command: ["python", "-m", "http.server"]
    secrets:
      HF_TOKEN: hf_leaked
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a list of secret names"):
        load_sidecars(cfg_path, environ={})


def test_load_sidecars_missing_secret_errors_unless_dry_run(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hf-jobs.yaml"
    cfg_path.write_text(
        """
sidecars:
  srv:
    image: python:3.12
    command: ["python", "-m", "http.server"]
    secrets: ["MISSING_SECRET"]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MISSING_SECRET"):
        load_sidecars(cfg_path, environ={}, dry_run=False)

    cfg = load_sidecars(cfg_path, environ={}, dry_run=True)
    assert cfg.sidecars["srv"].secrets == {"MISSING_SECRET": None}
    assert "MISSING_SECRET=<not set>" in format_sidecars_summary(cfg)


@pytest.mark.parametrize(
    "payload, error_match",
    [
        ({"sidecars": {"bad_name": {"image": "x", "command": ["a"]}}}, "Invalid sidecar name"),
        ({"sidecars": {"srv": {"image": "x", "command": ["a"], "unknown": 1}}}, "unknown key"),
        ({"sidecars": {"srv": {"image": "x"}}}, "must specify 'command'"),
        ({"sidecars": {"srv": {"command": ["a"]}}}, "must specify a non-empty 'image'"),
        ({"sidecars": {"srv": {"image": "x", "command": ["a"], "replicas": 0}}}, "replicas"),
        ({"sidecars": {"srv": {"image": "x", "command": ["a"], "replicas": True}}}, "replicas"),
        ({"sidecars": {"srv": {"image": "x", "command": ["a"], "replicas": 99}}}, "network group limit of 100"),
        (
            {"sidecars": {"srv": {"image": "x", "command": ["a"], "healthcheck": {"http": "8000/h", "tcp": 8000}}}},
            "exactly one of 'http' or 'tcp'",
        ),
        ({"sidecars": {"srv": {"image": "x", "command": ["a"], "on_exit": "restart"}}}, "on_exit"),
    ],
)
def test_load_sidecars_validation_errors(payload: dict[str, Any], error_match: str) -> None:
    with pytest.raises(ValueError, match=error_match):
        load_sidecars(payload, environ={})


def test_parse_main_job_marker() -> None:
    assert parse_main_job_marker("[sidecars] main_job_id=abc123 url=https://hf.co/jobs/u/abc123") == (
        "abc123",
        "https://hf.co/jobs/u/abc123",
    )
    assert parse_main_job_marker("[sidecars] started sidecar vllm") is None


# ---------------------------------------------------------------------------
# Orchestrator Job spec builder & secret isolation tests
# ---------------------------------------------------------------------------


def test_build_orchestrator_job_spec_isolates_secrets_and_sets_safety_timeouts() -> None:
    config = load_sidecars(
        {
            "defaults": {"startup_timeout": "2m", "readiness_timeout": "3m"},
            "sidecars": {
                "vllm": {
                    "image": "vllm/vllm-openai:latest",
                    "command": ["vllm", "serve", "m"],
                    "flavor": "a10g-small",
                    "secrets": {"HF_TOKEN": "hf_child_secret"},
                    "healthcheck": {"http": "8000/health"},
                }
            },
        }
    )
    main_spec = {
        "dockerImage": "python:3.12",
        "command": ["python", "run.py"],
        "arguments": [],
        "environment": {"FOO": "bar"},
        "secrets": {"MAIN_KEY": "main_secret_val"},
        "flavor": "cpu-basic",
        "timeoutSeconds": 600,
        "labels": {"name": "my-main"},
    }
    group_id, orch_spec = build_orchestrator_job_spec(
        sidecars_config=config,
        main_job_spec=main_spec,
        endpoint="https://huggingface.co",
        namespace="alice",
        orchestrator_token="hf_orch_token",
        resource_group_id="rg-1",
    )

    assert group_id.startswith("sc-")
    # Orchestrator token & child secrets are ONLY in `secrets`, never in `environment` or manifest.
    assert orch_spec["secrets"]["HF_ORCHESTRATOR_TOKEN"] == "hf_orch_token"
    secret_envelope = json.loads(orch_spec["secrets"]["HF_ORCHESTRATOR_SECRETS"])
    assert secret_envelope == {
        "main": {"MAIN_KEY": "main_secret_val"},
        "sidecars": {"vllm": {"HF_TOKEN": "hf_child_secret"}},
    }
    env_dump = json.dumps(orch_spec["environment"])
    assert "hf_orch_token" not in env_dump
    assert "hf_child_secret" not in env_dump
    assert "main_secret_val" not in env_dump

    manifest = json.loads(
        zlib.decompress(base64.b64decode(orch_spec["environment"]["HF_ORCHESTRATOR_MANIFEST"])).decode("utf-8")
    )
    assert manifest["group_id"] == group_id
    assert "secrets" not in manifest["main"]["spec"]
    assert "secrets" not in manifest["sidecars"][0]["spec"]
    assert manifest["main"]["spec"]["resourceGroupId"] == "rg-1"
    assert manifest["sidecars"][0]["spec"]["resourceGroupId"] == "rg-1"
    assert manifest["main"]["spec"]["labels"][SIDECAR_GROUP_LABEL] == group_id
    assert manifest["main"]["spec"]["labels"][SIDECAR_ROLE_LABEL] == "main"
    assert manifest["sidecars"][0]["spec"]["labels"][SIDECAR_NAME_LABEL] == "vllm"

    # Safety timeout floor: startup(120) + readiness(180) + main(600) + buffer(900) = 1800s.
    assert manifest["sidecars"][0]["spec"]["timeoutSeconds"] == 1800
    assert orch_spec["timeoutSeconds"] == 2700
    assert orch_spec["labels"] == {
        SIDECAR_GROUP_LABEL: group_id,
        SIDECAR_ROLE_LABEL: "orchestrator",
        "name": f"sidecars-orchestrator-{group_id[-6:]}",
    }


# ---------------------------------------------------------------------------
# End-to-end local orchestrator state machine tests
# ---------------------------------------------------------------------------


class _FakeHubState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counter = 0
        self.jobs: dict[str, dict[str, Any]] = {}
        self.created_specs: list[dict[str, Any]] = []
        self.cancelled_ids: list[str] = []
        self.logs_by_id: dict[str, list[str]] = {}
        self.on_create: Any = None
        self.on_list: Any = None


def _start_fake_hub(state: _FakeHubState) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            # POST /api/jobs/{ns}
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                with state.lock:
                    state.counter += 1
                    jid = f"job-{state.counter}"
                    state.created_specs.append(body)
                    stage = "RUNNING"
                    if state.on_create:
                        stage = state.on_create(jid, body)
                    job = {
                        "id": jid,
                        "labels": body.get("labels", {}),
                        "status": {
                            "stage": stage,
                            "message": None,
                            **(
                                {"exposeUrls": [f"https://{jid}--{p}.hf.jobs" for p in body["expose"]["ports"]]}
                                if body.get("expose") and body["expose"].get("ports")
                                else {}
                            ),
                        },
                    }
                    state.jobs[jid] = job
                payload = json.dumps(job).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                return

            # POST /api/jobs/{ns}/{id}/cancel
            if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[4] == "cancel":
                jid = parts[3]
                with state.lock:
                    state.cancelled_ids.append(jid)
                    if jid in state.jobs:
                        state.jobs[jid]["status"] = {"stage": "CANCELED", "message": "Canceled"}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
                return

            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            # GET /api/jobs/{ns}?label=...
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                with state.lock:
                    if state.on_list:
                        state.on_list(state)
                    items = list(state.jobs.values())
                payload = json.dumps(items).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                return

            # GET /api/jobs/{ns}/{id}/logs?tail=40
            if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[4] == "logs":
                jid = parts[3]
                with state.lock:
                    lines = list(state.logs_by_id.get(jid, []))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in lines:
                    ev = f"data: {json.dumps({'data': line})}\n\n".encode("utf-8")
                    self.wfile.write(ev)
                return

            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def _start_probe_http_server(fail_first: int = 0) -> tuple[ThreadingHTTPServer, int]:
    attempts = {"count": 0}

    class ProbeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            attempts["count"] += 1
            if self.path == "/health" and attempts["count"] > fail_first:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"loading")

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _make_manifest(endpoint: str, sidecars: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "namespace": "alice",
        "group_id": "sc-testgroup1",
        "group_label_key": SIDECAR_GROUP_LABEL,
        "sidecars": sidecars,
        "main": {
            "spec": {
                "dockerImage": "python:3.12",
                "command": ["python", "main.py"],
                "labels": {SIDECAR_GROUP_LABEL: "sc-testgroup1", SIDECAR_ROLE_LABEL: "main"},
            }
        },
    }


def test_orchestrator_happy_path_http_and_tcp_probes_and_teardown(capsys: pytest.CaptureFixture[str]) -> None:
    probe_http, http_port = _start_probe_http_server(fail_first=1)
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.bind(("127.0.0.1", 0))
    tcp_sock.listen(5)
    tcp_port = tcp_sock.getsockname()[1]

    state = _FakeHubState()
    polls_after_main = {"count": 0}

    def on_list(st: _FakeHubState) -> None:
        for job in st.jobs.values():
            if job["labels"].get(SIDECAR_ROLE_LABEL) == "main":
                polls_after_main["count"] += 1
                if polls_after_main["count"] >= 2:
                    job["status"] = {"stage": "COMPLETED", "message": None}

    state.on_list = on_list
    hub_srv, endpoint = _start_fake_hub(state)

    try:
        manifest = _make_manifest(
            endpoint,
            [
                {
                    "name": "vllm",
                    "service_name": "vllm",
                    "probe_alias": "vllm",
                    "startup_timeout_secs": 10,
                    "readiness_timeout_secs": 10,
                    "on_exit": "fail",
                    "healthcheck": {"kind": "http", "port": http_port, "path": "/health", "interval_secs": 0.05},
                    "spec": {
                        "dockerImage": "vllm",
                        "command": ["serve"],
                        "labels": {SIDECAR_GROUP_LABEL: "sc-testgroup1", SIDECAR_ROLE_LABEL: "sidecar"},
                    },
                },
                {
                    "name": "dask",
                    "service_name": "dask",
                    "probe_alias": "dask",
                    "startup_timeout_secs": 10,
                    "readiness_timeout_secs": 10,
                    "on_exit": "fail",
                    "healthcheck": {"kind": "tcp", "port": tcp_port, "interval_secs": 0.05},
                    "spec": {
                        "dockerImage": "dask",
                        "command": ["sched"],
                        "expose": {"ports": [8787], "portsPublic": []},
                        "labels": {SIDECAR_GROUP_LABEL: "sc-testgroup1", SIDECAR_ROLE_LABEL: "sidecar"},
                    },
                },
            ],
        )
        rc = run_orchestrator(
            manifest,
            token="hf_test",
            child_secrets={"sidecars": {"vllm": {"HF_TOKEN": "hf_model"}}, "main": {"MAIN_S": "1"}},
            probe_host_override="127.0.0.1",
            poll_interval_secs=0.05,
            install_signal_handlers=False,
        )
    finally:
        hub_srv.shutdown()
        probe_http.shutdown()
        tcp_sock.close()

    assert rc == 0
    # Sidecars were cancelled after main completed, main was NOT cancelled.
    main_ids = [jid for jid, j in state.jobs.items() if j["labels"].get(SIDECAR_ROLE_LABEL) == "main"]
    sidecar_ids = [jid for jid, j in state.jobs.items() if j["labels"].get(SIDECAR_ROLE_LABEL) == "sidecar"]
    assert len(main_ids) == 1
    assert set(state.cancelled_ids) == set(sidecar_ids)
    # Secrets were forwarded into the right child job specs.
    vllm_spec = next(s for s in state.created_specs if s["dockerImage"] == "vllm")
    main_spec = next(s for s in state.created_specs if s["dockerImage"] == "python:3.12")
    assert vllm_spec["secrets"] == {"HF_TOKEN": "hf_model"}
    assert main_spec["secrets"] == {"MAIN_S": "1"}
    out = capsys.readouterr().out
    assert f"[sidecars] main_job_id={main_ids[0]}" in out
    assert "--8787.hf.jobs" in out


def test_orchestrator_fails_fast_on_mid_run_sidecar_death_and_dumps_logs(capsys: pytest.CaptureFixture[str]) -> None:
    state = _FakeHubState()
    polls_after_main = {"count": 0}

    def on_list(st: _FakeHubState) -> None:
        has_main = any(j["labels"].get(SIDECAR_ROLE_LABEL) == "main" for j in st.jobs.values())
        if has_main:
            polls_after_main["count"] += 1
            if polls_after_main["count"] == 1:
                for jid, j in st.jobs.items():
                    if j["labels"].get(SIDECAR_NAME_LABEL) == "worker-0":
                        j["status"] = {"stage": "ERROR", "message": "OOMKilled (exit 137)"}
                        st.logs_by_id[jid] = ["loading batch", "MemoryError: OOM"]

    state.on_list = on_list
    hub_srv, endpoint = _start_fake_hub(state)

    try:
        manifest = _make_manifest(
            endpoint,
            [
                {
                    "name": "worker-0",
                    "service_name": "worker",
                    "probe_alias": "worker-0",
                    "startup_timeout_secs": 5,
                    "readiness_timeout_secs": 5,
                    "on_exit": "fail",
                    "healthcheck": None,
                    "spec": {
                        "dockerImage": "img",
                        "command": ["run"],
                        "labels": {
                            SIDECAR_GROUP_LABEL: "sc-testgroup1",
                            SIDECAR_ROLE_LABEL: "sidecar",
                            SIDECAR_NAME_LABEL: "worker-0",
                        },
                    },
                },
                {
                    "name": "worker-1",
                    "service_name": "worker",
                    "probe_alias": "worker-1",
                    "startup_timeout_secs": 5,
                    "readiness_timeout_secs": 5,
                    "on_exit": "fail",
                    "healthcheck": None,
                    "spec": {
                        "dockerImage": "img",
                        "command": ["run"],
                        "labels": {
                            SIDECAR_GROUP_LABEL: "sc-testgroup1",
                            SIDECAR_ROLE_LABEL: "sidecar",
                            SIDECAR_NAME_LABEL: "worker-1",
                        },
                    },
                },
            ],
        )
        rc = run_orchestrator(
            manifest,
            token="hf_test",
            probe_host_override="127.0.0.1",
            poll_interval_secs=0.05,
            install_signal_handlers=False,
        )
    finally:
        hub_srv.shutdown()

    assert rc == 1
    # Both main and healthy sibling worker-1 were cancelled.
    expected_cancelled = {jid for jid, j in state.jobs.items() if j["labels"].get(SIDECAR_NAME_LABEL) != "worker-0"}
    assert set(state.cancelled_ids) == expected_cancelled
    out = capsys.readouterr().out
    assert "OOMKilled (exit 137)" in out
    assert "MemoryError: OOM" in out


def test_orchestrator_readiness_timeout_cancels_sidecar_before_main(capsys: pytest.CaptureFixture[str]) -> None:
    probe_http, http_port = _start_probe_http_server(fail_first=999)
    state = _FakeHubState()
    hub_srv, endpoint = _start_fake_hub(state)
    try:
        manifest = _make_manifest(
            endpoint,
            [
                {
                    "name": "vllm",
                    "service_name": "vllm",
                    "probe_alias": "vllm",
                    "startup_timeout_secs": 5,
                    "readiness_timeout_secs": 0.15,
                    "on_exit": "fail",
                    "healthcheck": {"kind": "http", "port": http_port, "path": "/health", "interval_secs": 0.05},
                    "spec": {
                        "dockerImage": "vllm",
                        "command": ["serve"],
                        "labels": {SIDECAR_GROUP_LABEL: "sc-testgroup1", SIDECAR_ROLE_LABEL: "sidecar"},
                    },
                }
            ],
        )
        rc = run_orchestrator(
            manifest,
            token="hf_test",
            probe_host_override="127.0.0.1",
            poll_interval_secs=0.05,
            install_signal_handlers=False,
        )
    finally:
        hub_srv.shutdown()
        probe_http.shutdown()

    assert rc == 1
    assert len(state.created_specs) == 1  # main was never created
    assert state.cancelled_ids == ["job-1"]
    assert "failed readiness check" in capsys.readouterr().out


def test_orchestrator_startup_timeout_cancels_scheduling_sidecar(capsys: pytest.CaptureFixture[str]) -> None:
    state = _FakeHubState()
    state.on_create = lambda _jid, _body: "SCHEDULING"
    hub_srv, endpoint = _start_fake_hub(state)
    try:
        manifest = _make_manifest(
            endpoint,
            [
                {
                    "name": "gpu-srv",
                    "service_name": "gpu-srv",
                    "probe_alias": "gpu-srv",
                    "startup_timeout_secs": 0.12,
                    "readiness_timeout_secs": 5,
                    "on_exit": "fail",
                    "healthcheck": None,
                    "spec": {
                        "dockerImage": "vllm",
                        "command": ["serve"],
                        "labels": {SIDECAR_GROUP_LABEL: "sc-testgroup1", SIDECAR_ROLE_LABEL: "sidecar"},
                    },
                }
            ],
        )
        rc = run_orchestrator(
            manifest,
            token="hf_test",
            probe_host_override="127.0.0.1",
            poll_interval_secs=0.05,
            install_signal_handlers=False,
        )
    finally:
        hub_srv.shutdown()

    assert rc == 1
    assert len(state.created_specs) == 1
    assert state.cancelled_ids == ["job-1"]
    assert "timed out waiting to start" in capsys.readouterr().out


def test_orchestrator_catches_already_healthy_sidecar_crash_during_startup_and_logs_heartbeat(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _FakeHubState()
    polls = {"count": 0}

    def on_create(_jid: str, body: dict[str, Any]) -> str:
        return "RUNNING" if body["labels"].get(SIDECAR_NAME_LABEL) == "fast" else "SCHEDULING"

    def on_list(st: _FakeHubState) -> None:
        polls["count"] += 1
        if polls["count"] >= 3:
            for jid, j in st.jobs.items():
                if j["labels"].get(SIDECAR_NAME_LABEL) == "fast":
                    j["status"] = {"stage": "ERROR", "message": "OOMKilled"}
                    st.logs_by_id[jid] = ["fast died while slow was scheduling"]

    state.on_create = on_create
    state.on_list = on_list
    hub_srv, endpoint = _start_fake_hub(state)
    try:
        manifest = _make_manifest(
            endpoint,
            [
                {
                    "name": "fast",
                    "service_name": "fast",
                    "probe_alias": "fast",
                    "startup_timeout_secs": 10,
                    "readiness_timeout_secs": 10,
                    "on_exit": "fail",
                    "healthcheck": None,
                    "spec": {
                        "dockerImage": "img",
                        "command": ["run"],
                        "labels": {
                            SIDECAR_GROUP_LABEL: "sc-testgroup1",
                            SIDECAR_ROLE_LABEL: "sidecar",
                            SIDECAR_NAME_LABEL: "fast",
                        },
                    },
                },
                {
                    "name": "slow",
                    "service_name": "slow",
                    "probe_alias": "slow",
                    "startup_timeout_secs": 10,
                    "readiness_timeout_secs": 10,
                    "on_exit": "fail",
                    "healthcheck": None,
                    "spec": {
                        "dockerImage": "img",
                        "command": ["run"],
                        "labels": {
                            SIDECAR_GROUP_LABEL: "sc-testgroup1",
                            SIDECAR_ROLE_LABEL: "sidecar",
                            SIDECAR_NAME_LABEL: "slow",
                        },
                    },
                },
            ],
        )
        rc = run_orchestrator(
            manifest,
            token="hf_test",
            probe_host_override="127.0.0.1",
            poll_interval_secs=0.05,
            heartbeat_interval_secs=0.04,
            install_signal_handlers=False,
        )
    finally:
        hub_srv.shutdown()

    assert rc == 1
    assert len(state.created_specs) == 2  # main was never created
    slow_id = next(jid for jid, j in state.jobs.items() if j["labels"].get(SIDECAR_NAME_LABEL) == "slow")
    assert state.cancelled_ids == [slow_id]
    out = capsys.readouterr().out
    assert "waiting (" in out
    assert "fast=healthy, slow=SCHEDULING" in out
    assert "exited while waiting for other sidecars" in out
    assert "fast died while slow was scheduling" in out


# ---------------------------------------------------------------------------
# HfApi.run_job & CLI integration tests
# ---------------------------------------------------------------------------


def test_run_job_with_sidecars_rejects_token_false() -> None:
    api = HfApi(token="hf_stored_token")
    with (
        patch.object(api, "whoami", return_value={"name": "alice"}),
        patch("huggingface_hub.constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING", False),
        pytest.warns(UserWarning, match="experimental"),
        pytest.raises(ValueError, match="token=False"),
    ):
        api.run_job(
            image="python:3.12",
            command=["echo", "hi"],
            sidecars={"srv": {"image": "python:3.12", "command": ["python", "-m", "http.server"]}},
            token=False,
        )


def test_run_job_with_sidecars_submits_orchestrator() -> None:
    api = HfApi(token="hf_test_token")
    fake_response = Mock()
    fake_response.json.return_value = {
        "id": "orch-123",
        "dockerImage": "python:3.12-slim",
        "command": ["python3", "-c", "..."],
        "arguments": [],
        "environment": {},
        "secrets": {},
        "flavor": "cpu-basic",
        "labels": {SIDECAR_GROUP_LABEL: "sc-abc", SIDECAR_ROLE_LABEL: "orchestrator"},
        "status": {"stage": "SCHEDULING"},
        "owner": {"id": "u1", "name": "alice", "type": "user"},
    }
    with (
        patch.object(api, "whoami", return_value={"name": "alice"}),
        patch("huggingface_hub.constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING", False),
        patch("huggingface_hub.hf_api.get_session") as mock_session,
        patch("huggingface_hub.hf_api.hf_raise_for_status"),
        pytest.warns(UserWarning, match="experimental"),
    ):
        mock_session.return_value.post.return_value = fake_response
        job = api.run_job(
            image="python:3.12",
            command=["python", "client.py"],
            sidecars={"srv": {"image": "python:3.12", "command": ["python", "-m", "http.server", "8000"]}},
        )

    assert job.id == "orch-123"
    posted = mock_session.return_value.post.call_args.kwargs["json"]
    assert posted["dockerImage"] == "python:3.12-slim"
    assert posted["secrets"]["HF_ORCHESTRATOR_TOKEN"] == "hf_test_token"
    assert "HF_ORCHESTRATOR_TOKEN" not in posted["environment"]


def test_cli_run_sidecars_dry_run(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hf-jobs.yaml"
    cfg_path.write_text(
        """
sidecars:
  vllm:
    image: vllm/vllm-openai:latest
    command: ["vllm", "serve", "m"]
    flavor: a10g-small
    secrets: ["HF_TOKEN"]
    healthcheck:
      http: "8000/health"
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    with (
        patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
        patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
    ):
        result = runner.invoke(
            app,
            ["jobs", "run", "--dry-run", "--sidecars", str(cfg_path), "python:3.12", "python", "client.py"],
        )
    assert result.exit_code == 0
    assert "vllm (flavor=a10g-small, health=http:8000/health, secrets=[HF_TOKEN=<not set>])" in result.output
    assert "(dry run) Job not submitted." in result.output
    api_cls.return_value.run_job.assert_not_called()


def test_cli_run_sidecars_attached_hands_off_to_main_logs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hf-jobs.yaml"
    cfg_path.write_text(
        """
sidecars:
  srv:
    image: python:3.12
    command: ["python", "-m", "http.server", "8000"]
""",
        encoding="utf-8",
    )
    owner = JobOwner(id="u1", name="alice", type="user")
    orch_job = Mock(
        id="orch-1",
        owner=owner,
        url="https://huggingface.co/jobs/alice/orch-1",
        labels={SIDECAR_GROUP_LABEL: "sc-123", SIDECAR_ROLE_LABEL: "orchestrator", "name": "sidecars-orchestrator"},
    )
    orch_job.status.stage = JobStage.RUNNING
    main_job = Mock(
        id="main-1",
        owner=owner,
        url="https://huggingface.co/jobs/alice/main-1",
        labels={SIDECAR_GROUP_LABEL: "sc-123", SIDECAR_ROLE_LABEL: "main"},
    )
    main_job.status.stage = JobStage.RUNNING
    main_final = Mock(id="main-1")
    main_final.status.stage = JobStage.COMPLETED
    main_final.status.message = None

    runner = CliRunner()
    with (
        patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
        patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
    ):
        api = api_cls.return_value
        api.run_job.return_value = orch_job
        api.inspect_job.return_value = main_job
        api.fetch_job_logs.side_effect = [
            iter(
                [
                    "[sidecars] started sidecar 'srv'",
                    "[sidecars] main_job_id=main-1 url=https://hf.co/jobs/alice/main-1",
                ]
            ),
            iter(["hello from main"]),
        ]
        api.wait_for_job.return_value = main_final
        result = runner.invoke(
            app,
            ["jobs", "run", "--sidecars", str(cfg_path), "python:3.12", "python", "client.py"],
        )

    assert result.exit_code == 0
    assert "started sidecar 'srv'" in result.output
    assert "hello from main" in result.output
    assert "Job main-1 completed" in result.output
    assert api.fetch_job_logs.call_count == 2
    assert api.fetch_job_logs.call_args_list[0].kwargs["job_id"] == "orch-1"
    assert api.fetch_job_logs.call_args_list[1].kwargs["job_id"] == "main-1"


def test_cli_run_sidecars_orchestrator_failure_raises_cli_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hf-jobs.yaml"
    cfg_path.write_text(
        """
sidecars:
  srv:
    image: python:3.12
    command: ["python", "-m", "http.server", "8000"]
""",
        encoding="utf-8",
    )
    owner = JobOwner(id="u1", name="alice", type="user")
    orch_job = Mock(
        id="orch-1",
        owner=owner,
        url="https://huggingface.co/jobs/alice/orch-1",
        labels={SIDECAR_GROUP_LABEL: "sc-123", SIDECAR_ROLE_LABEL: "orchestrator"},
    )
    orch_job.status.stage = JobStage.RUNNING
    orch_final = Mock(id="orch-1")
    orch_final.status.stage = JobStage.ERROR
    orch_final.status.message = "exit code 1"

    runner = CliRunner()
    with (
        patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
        patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
    ):
        api = api_cls.return_value
        api.run_job.return_value = orch_job
        api.fetch_job_logs.return_value = iter(["[sidecars] ERROR: sidecar 'srv' failed readiness check"])
        api.wait_for_job.return_value = orch_final
        result = runner.invoke(
            app,
            ["jobs", "run", "--sidecars", str(cfg_path), "python:3.12", "python", "client.py"],
        )

    assert result.exit_code == 1
    assert isinstance(result.exception, CLIError)
    assert "finished with stage 'ERROR'" in str(result.exception)
