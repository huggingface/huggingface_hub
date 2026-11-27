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
"""Client-side sidecar config parsing, validation, and orchestrator Job builder for HF Jobs."""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
import uuid
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from huggingface_hub._jobs_api import JobHardware, _create_job_spec, _parse_job_timeout, _sanitize_job_name
from huggingface_hub._jobs_sidecar_orchestrator import MAIN_JOB_ID_PREFIX
from huggingface_hub._space_api import Volume
from huggingface_hub.errors import HfUriError
from huggingface_hub.utils._auth import get_token
from huggingface_hub.utils._parsing import parse_duration


SIDECAR_GROUP_LABEL = "hf-sidecar-group"
SIDECAR_ROLE_LABEL = "hf-sidecar-role"
SIDECAR_NAME_LABEL = "hf-sidecar-name"

ORCHESTRATOR_IMAGE = "python:3.12-slim"
DEFAULT_JOB_TIMEOUT_SECS = 30 * 60
DEFAULT_STARTUP_TIMEOUT_SECS = 5 * 60
DEFAULT_READINESS_TIMEOUT_SECS = 10 * 60
DEFAULT_HEALTH_INTERVAL_SECS = 2.0
DEFAULT_HEALTH_PROBE_TIMEOUT_SECS = 5.0
SAFETY_TIMEOUT_BUFFER_SECS = 5 * 60

# Hub network groups support at most 100 members (sidecars + main + orchestrator).
MAX_NETWORK_GROUP_MEMBERS = 100
_MAX_ALIAS_LENGTH = 34
_MAX_GROUP_LENGTH = 46
_ALIAS_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,32}[a-z0-9])?$")
_GROUP_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?$")
_MAIN_JOB_MARKER_RE = re.compile(r"^\[sidecars\] main_job_id=(\S+)(?: url=(\S+))?$")

_VALID_TOP_LEVEL_KEYS = ("defaults", "sidecars")
_VALID_DEFAULTS_KEYS = ("startup_timeout", "readiness_timeout", "on_exit")
_VALID_SIDECAR_KEYS = (
    "image",
    "command",
    "flavor",
    "replicas",
    "env",
    "secrets",
    "volumes",
    "expose",
    "expose_public",
    "ssh",
    "labels",
    "timeout",
    "healthcheck",
    "startup_timeout",
    "readiness_timeout",
    "on_exit",
)
_VALID_HEALTHCHECK_KEYS = ("http", "tcp", "interval", "timeout")
_VALID_ON_EXIT_VALUES = ("fail", "ignore")


@dataclass(frozen=True)
class SidecarHealthCheck:
    kind: Literal["http", "tcp"]
    port: int
    path: str = "/"
    interval_secs: float = DEFAULT_HEALTH_INTERVAL_SECS
    probe_timeout_secs: float = DEFAULT_HEALTH_PROBE_TIMEOUT_SECS

    def to_manifest_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "port": self.port,
            "interval_secs": self.interval_secs,
            "probe_timeout_secs": self.probe_timeout_secs,
        }
        if self.kind == "http":
            data["path"] = self.path
        return data

    def format_short(self) -> str:
        if self.kind == "http":
            return f"http:{self.port}{self.path}"
        return f"tcp:{self.port}"


@dataclass(frozen=True)
class SidecarSpec:
    name: str
    image: str
    command: list[str]
    flavor: JobHardware | str | None = None
    replicas: int = 1
    env: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str | None] = field(default_factory=dict)
    volumes: list[Volume] = field(default_factory=list)
    expose: list[int] | None = None
    expose_public: list[int] | None = None
    ssh: bool = False
    labels: dict[str, str] = field(default_factory=dict)
    timeout_secs: int | None = None
    healthcheck: SidecarHealthCheck | None = None
    startup_timeout_secs: int = DEFAULT_STARTUP_TIMEOUT_SECS
    readiness_timeout_secs: int = DEFAULT_READINESS_TIMEOUT_SECS
    on_exit: Literal["fail", "ignore"] = "fail"


@dataclass(frozen=True)
class SidecarInstance:
    """One concrete Job expanded from a `SidecarSpec` (handling `replicas > 1`)."""

    instance_name: str
    service_name: str
    network_aliases: list[str]
    probe_alias: str
    spec: SidecarSpec


@dataclass(frozen=True)
class SidecarsConfig:
    sidecars: dict[str, SidecarSpec]
    source_label: str | None = None

    @property
    def total_instances(self) -> int:
        return sum(spec.replicas for spec in self.sidecars.values())

    def expand_instances(self) -> list[SidecarInstance]:
        instances: list[SidecarInstance] = []
        for name, spec in self.sidecars.items():
            if spec.replicas == 1:
                instances.append(
                    SidecarInstance(
                        instance_name=name,
                        service_name=name,
                        network_aliases=[name],
                        probe_alias=name,
                        spec=spec,
                    )
                )
            else:
                for idx in range(spec.replicas):
                    inst_name = f"{name}-{idx}"
                    instances.append(
                        SidecarInstance(
                            instance_name=inst_name,
                            service_name=name,
                            network_aliases=[name, inst_name],
                            probe_alias=inst_name,
                            spec=spec,
                        )
                    )
        return instances


def _parse_positive_duration_secs(value: int | float | str, *, field_name: str) -> float:
    """Parse a positive duration (`300`, `5m`, `1.5h`, `0.5s`) into seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"'{field_name}' must be a duration string (e.g. '5m') or number of seconds, got {value!r}.")
    if isinstance(value, (int, float)):
        secs = float(value)
    else:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"'{field_name}' cannot be empty.")
        try:
            # Handle fractional units (e.g. "0.5s", "1.5h") and standard `parse_duration` units.
            factors = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
            if stripped[-1] in factors:
                secs = float(stripped[:-1]) * factors[stripped[-1]]
            else:
                secs = float(parse_duration(stripped))
        except ValueError:
            try:
                secs = float(_parse_job_timeout(stripped))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid '{field_name}' value {value!r}. Use seconds or a duration like '30s', '5m', '2h'."
                ) from exc
    if secs <= 0:
        raise ValueError(f"'{field_name}' must be positive, got {value!r}.")
    return secs


def parse_timeout_secs(value: int | float | str, *, field_name: str = "timeout") -> int:
    """Parse a positive duration (`300`, `5m`, `1.5h`) into integer seconds."""
    secs = int(_parse_positive_duration_secs(value, field_name=field_name))
    if secs <= 0:
        raise ValueError(f"'{field_name}' must be at least 1 second, got {value!r}.")
    return secs


def _validate_alias(alias: str, *, context: str) -> None:
    if not isinstance(alias, str) or not _ALIAS_RE.fullmatch(alias):
        raise ValueError(
            f"Invalid {context} '{alias}'. Must be lowercase alphanumerics and dashes, 1-{_MAX_ALIAS_LENGTH} chars, "
            "and cannot start or end with a dash."
        )


def _validate_network_group(group: str) -> None:
    if not isinstance(group, str) or not _GROUP_RE.fullmatch(group):
        raise ValueError(
            f"Invalid network_group '{group}'. Must be lowercase alphanumerics and dashes, 1-{_MAX_GROUP_LENGTH} chars."
        )


def _parse_port(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 65535):
        raise ValueError(f"'{field_name}' must be a TCP port integer between 1 and 65535, got {value!r}.")
    return value


def _parse_healthcheck(raw: Any, *, sidecar_name: str) -> SidecarHealthCheck | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"Sidecar '{sidecar_name}': 'healthcheck' must be a mapping (e.g. {{http: '8000/health'}}).")
    unknown = sorted(k for k in raw if k not in _VALID_HEALTHCHECK_KEYS)
    if unknown:
        raise ValueError(
            f"Sidecar '{sidecar_name}': unknown healthcheck key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Valid keys: {', '.join(_VALID_HEALTHCHECK_KEYS)}."
        )

    has_http = raw.get("http") is not None
    has_tcp = raw.get("tcp") is not None
    if has_http == has_tcp:
        raise ValueError(f"Sidecar '{sidecar_name}': 'healthcheck' must define exactly one of 'http' or 'tcp'.")

    interval_secs = (
        _parse_positive_duration_secs(raw["interval"], field_name=f"{sidecar_name}.healthcheck.interval")
        if raw.get("interval") is not None
        else DEFAULT_HEALTH_INTERVAL_SECS
    )
    probe_timeout_secs = (
        _parse_positive_duration_secs(raw["timeout"], field_name=f"{sidecar_name}.healthcheck.timeout")
        if raw.get("timeout") is not None
        else DEFAULT_HEALTH_PROBE_TIMEOUT_SECS
    )

    if has_tcp:
        port = _parse_port(raw["tcp"], field_name=f"{sidecar_name}.healthcheck.tcp")
        return SidecarHealthCheck(
            kind="tcp", port=port, interval_secs=interval_secs, probe_timeout_secs=probe_timeout_secs
        )

    http_raw = raw["http"]
    if isinstance(http_raw, Mapping):
        if "port" not in http_raw:
            raise ValueError(f"Sidecar '{sidecar_name}': 'healthcheck.http' mapping must contain 'port'.")
        port = _parse_port(http_raw["port"], field_name=f"{sidecar_name}.healthcheck.http.port")
        path = str(http_raw.get("path", "/"))
    elif isinstance(http_raw, str):
        cleaned = http_raw.strip().lstrip(":")
        if "/" in cleaned:
            port_str, rest = cleaned.split("/", 1)
            path = "/" + rest
        else:
            port_str, path = cleaned, "/"
        try:
            port_int = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Sidecar '{sidecar_name}': invalid 'healthcheck.http' value {http_raw!r}. "
                "Use '<port>/<path>' (e.g. '8000/health') or {{port: 8000, path: '/health'}}."
            ) from exc
        port = _parse_port(port_int, field_name=f"{sidecar_name}.healthcheck.http")
    else:
        raise ValueError(
            f"Sidecar '{sidecar_name}': 'healthcheck.http' must be a string (e.g. '8000/health') or mapping."
        )

    if not path.startswith("/"):
        path = "/" + path
    return SidecarHealthCheck(
        kind="http",
        port=port,
        path=path,
        interval_secs=interval_secs,
        probe_timeout_secs=probe_timeout_secs,
    )


def _parse_sidecar_volumes(raw: Any, *, sidecar_name: str) -> list[Volume]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Sidecar '{sidecar_name}': 'volumes' must be a list of hf:// mount specs or Volume objects.")
    volumes: list[Volume] = []
    for entry in raw:
        if isinstance(entry, Volume):
            volumes.append(entry)
        elif isinstance(entry, str):
            try:
                volumes.append(Volume.from_uri(entry))
            except HfUriError as exc:
                raise ValueError(f"Sidecar '{sidecar_name}': invalid volume '{entry}': {exc}") from exc
        else:
            raise ValueError(
                f"Sidecar '{sidecar_name}': volume entries must be hf:// strings or Volume objects, got {entry!r}."
            )
    return volumes


def _resolve_sidecar_secrets(
    raw: Any,
    *,
    sidecar_name: str,
    environ: Mapping[str, str],
    allow_secret_values: bool,
    dry_run: bool,
) -> dict[str, str | None]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        if not allow_secret_values:
            raise ValueError(
                f"Sidecar '{sidecar_name}': 'secrets' in a YAML file must be a list of secret names "
                '(e.g. `secrets: ["HF_TOKEN"]`), not key/value pairs: secret values are always read from the local environment.'
            )
        resolved: dict[str, str | None] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or (v is not None and not isinstance(v, str)):
                raise ValueError(f"Sidecar '{sidecar_name}': 'secrets' keys and values must be strings.")
            resolved[k] = v
        return resolved
    if isinstance(raw, list):
        resolved = {}
        missing: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Sidecar '{sidecar_name}': each secret name must be a non-empty string.")
            secret_name = item.strip()
            if secret_name in environ:
                resolved[secret_name] = environ[secret_name]
            else:
                missing.append(secret_name)
                resolved[secret_name] = None
        if missing and not dry_run:
            raise ValueError(
                f"Sidecar '{sidecar_name}' requires secret(s) not set in your environment: {', '.join(missing)}. "
                f"Export them locally (e.g. `export {missing[0]}=...`) before running."
            )
        return resolved
    raise ValueError(f"Sidecar '{sidecar_name}': 'secrets' must be a list of environment variable names.")


def _default_environ() -> dict[str, str]:
    env = dict(os.environ)
    if "HF_TOKEN" not in env:
        stored = get_token()
        if stored:
            env["HF_TOKEN"] = stored
    return env


def load_sidecars(
    source: SidecarsConfig | Path | str | Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    allow_secret_values: bool | None = None,
    dry_run: bool = False,
) -> SidecarsConfig:
    """Load and validate sidecar definitions from an `hf-jobs.yaml` path or Python dict."""
    if isinstance(source, SidecarsConfig):
        return source

    source_label: str | None = None
    if isinstance(source, (Path, str)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Sidecars config file not found: '{source}'.")
        source_label = str(path)
        try:
            raw_doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in sidecars config '{source}': {exc}") from exc
        if allow_secret_values is None:
            allow_secret_values = False
    elif isinstance(source, Mapping):
        raw_doc = source
        if allow_secret_values is None:
            allow_secret_values = True
    else:
        raise ValueError(
            f"'sidecars' must be a path to a YAML file (e.g. 'hf-jobs.yaml') or a dict, got {type(source).__name__}."
        )

    if not isinstance(raw_doc, Mapping) or not raw_doc:
        raise ValueError("Sidecars config must be a non-empty mapping.")

    # Support both:
    #   1. Full document: {"defaults": {...}, "sidecars": {"vllm": {...}}}
    #   2. Shorthand dict (Python API): {"vllm-1": {"image": ..., "command": ...}}
    is_full_doc = "sidecars" in raw_doc and (
        not isinstance(raw_doc["sidecars"], Mapping) or "image" not in raw_doc["sidecars"]
    )
    if is_full_doc:
        unknown_top = sorted(k for k in raw_doc if k not in _VALID_TOP_LEVEL_KEYS)
        if unknown_top:
            raise ValueError(
                f"Unknown top-level key(s) {', '.join(repr(k) for k in unknown_top)} in sidecars config. "
                f"Valid keys: {', '.join(_VALID_TOP_LEVEL_KEYS)}."
            )
        defaults_raw = raw_doc.get("defaults") or {}
        sidecars_raw = raw_doc.get("sidecars")
    else:
        if source_label is not None:
            raise ValueError(f"Sidecars file '{source_label}' must contain a top-level 'sidecars' section.")
        defaults_raw = {}
        sidecars_raw = raw_doc

    if not isinstance(defaults_raw, Mapping):
        raise ValueError("'defaults' in sidecars config must be a mapping.")
    unknown_defaults = sorted(k for k in defaults_raw if k not in _VALID_DEFAULTS_KEYS)
    if unknown_defaults:
        raise ValueError(
            f"Unknown key(s) {', '.join(repr(k) for k in unknown_defaults)} in 'defaults'. "
            f"Valid keys: {', '.join(_VALID_DEFAULTS_KEYS)}."
        )

    default_startup = (
        parse_timeout_secs(defaults_raw["startup_timeout"], field_name="defaults.startup_timeout")
        if defaults_raw.get("startup_timeout") is not None
        else DEFAULT_STARTUP_TIMEOUT_SECS
    )
    default_readiness = (
        parse_timeout_secs(defaults_raw["readiness_timeout"], field_name="defaults.readiness_timeout")
        if defaults_raw.get("readiness_timeout") is not None
        else DEFAULT_READINESS_TIMEOUT_SECS
    )
    default_on_exit = defaults_raw.get("on_exit", "fail")
    if default_on_exit not in _VALID_ON_EXIT_VALUES:
        raise ValueError(f"'defaults.on_exit' must be one of {_VALID_ON_EXIT_VALUES}, got {default_on_exit!r}.")

    if not isinstance(sidecars_raw, Mapping) or not sidecars_raw:
        raise ValueError("'sidecars' must be a non-empty mapping of sidecar definitions.")

    resolved_env = environ if environ is not None else _default_environ()
    specs: dict[str, SidecarSpec] = {}

    for name, entry in sidecars_raw.items():
        _validate_alias(str(name), context="sidecar name")
        if not isinstance(entry, Mapping):
            raise ValueError(f"Sidecar '{name}' must be a mapping.")
        unknown = sorted(k for k in entry if k not in _VALID_SIDECAR_KEYS)
        if unknown:
            raise ValueError(
                f"Sidecar '{name}': unknown key(s) {', '.join(repr(k) for k in unknown)}. "
                f"Valid keys: {', '.join(_VALID_SIDECAR_KEYS)}."
            )

        image = entry.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"Sidecar '{name}' must specify a non-empty 'image' string.")

        raw_cmd = entry.get("command")
        if isinstance(raw_cmd, str) and raw_cmd.strip():
            command = ["sh", "-c", raw_cmd]
        elif isinstance(raw_cmd, list) and len(raw_cmd) > 0 and all(isinstance(part, str) for part in raw_cmd):
            command = list(raw_cmd)
        else:
            raise ValueError(f"Sidecar '{name}' must specify 'command' as a non-empty string or list of strings.")

        replicas = entry.get("replicas", 1)
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
            raise ValueError(f"Sidecar '{name}': 'replicas' must be an integer >= 1, got {replicas!r}.")
        if replicas > 1:
            _validate_alias(f"{name}-{replicas - 1}", context=f"sidecar '{name}' replica alias")

        flavor = entry.get("flavor")
        if flavor is not None and not isinstance(flavor, (str, JobHardware)):
            raise ValueError(f"Sidecar '{name}': 'flavor' must be a string, got {flavor!r}.")

        env_raw = entry.get("env") or {}
        if not isinstance(env_raw, Mapping):
            raise ValueError(f"Sidecar '{name}': 'env' must be a mapping of string keys to values.")
        env_map: dict[str, str] = {}
        for ek, ev in env_raw.items():
            if not isinstance(ek, str):
                raise ValueError(f"Sidecar '{name}': 'env' keys must be strings.")
            env_map[ek] = "" if ev is None else str(ev)

        labels_raw = entry.get("labels") or {}
        if not isinstance(labels_raw, Mapping) or any(
            not isinstance(lk, str) or not isinstance(lv, str) for lk, lv in labels_raw.items()
        ):
            raise ValueError(f"Sidecar '{name}': 'labels' must be a mapping of string keys to string values.")

        secrets_map = _resolve_sidecar_secrets(
            entry.get("secrets"),
            sidecar_name=name,
            environ=resolved_env,
            allow_secret_values=allow_secret_values,
            dry_run=dry_run,
        )
        volumes = _parse_sidecar_volumes(entry.get("volumes"), sidecar_name=name)

        expose = entry.get("expose")
        if expose is not None:
            if not isinstance(expose, list):
                raise ValueError(f"Sidecar '{name}': 'expose' must be a list of port integers.")
            expose = [_parse_port(p, field_name=f"{name}.expose") for p in expose]

        expose_public = entry.get("expose_public")
        if expose_public is not None:
            if not isinstance(expose_public, list):
                raise ValueError(f"Sidecar '{name}': 'expose_public' must be a list of port integers.")
            expose_public = [_parse_port(p, field_name=f"{name}.expose_public") for p in expose_public]

        ssh = entry.get("ssh", False)
        if not isinstance(ssh, bool):
            raise ValueError(f"Sidecar '{name}': 'ssh' must be a boolean.")

        timeout_secs = (
            parse_timeout_secs(entry["timeout"], field_name=f"{name}.timeout")
            if entry.get("timeout") is not None
            else None
        )
        startup_timeout_secs = (
            parse_timeout_secs(entry["startup_timeout"], field_name=f"{name}.startup_timeout")
            if entry.get("startup_timeout") is not None
            else default_startup
        )
        readiness_timeout_secs = (
            parse_timeout_secs(entry["readiness_timeout"], field_name=f"{name}.readiness_timeout")
            if entry.get("readiness_timeout") is not None
            else default_readiness
        )
        on_exit = entry.get("on_exit", default_on_exit)
        if on_exit not in _VALID_ON_EXIT_VALUES:
            raise ValueError(f"Sidecar '{name}': 'on_exit' must be one of {_VALID_ON_EXIT_VALUES}, got {on_exit!r}.")

        healthcheck = _parse_healthcheck(entry.get("healthcheck"), sidecar_name=name)

        specs[name] = SidecarSpec(
            name=name,
            image=image,
            command=command,
            flavor=flavor,
            replicas=replicas,
            env=env_map,
            secrets=secrets_map,
            volumes=volumes,
            expose=expose,
            expose_public=expose_public,
            ssh=ssh,
            labels=dict(labels_raw),
            timeout_secs=timeout_secs,
            healthcheck=healthcheck,
            startup_timeout_secs=startup_timeout_secs,
            readiness_timeout_secs=readiness_timeout_secs,
            on_exit=on_exit,
        )

    config = SidecarsConfig(sidecars=specs, source_label=source_label)
    # Hub network groups cap total members at 100 (sidecars + main + orchestrator).
    total_members = config.total_instances + 2
    if total_members > MAX_NETWORK_GROUP_MEMBERS:
        raise ValueError(
            f"Sidecar run would create {total_members} Jobs ({config.total_instances} sidecar(s) + main + orchestrator), "
            f"exceeding the Hub network group limit of {MAX_NETWORK_GROUP_MEMBERS} members."
        )
    return config


def format_sidecars_summary(config: SidecarsConfig) -> str:
    """Format a compact multi-line summary of sidecars for CLI run/dry-run output."""
    lines: list[str] = []
    for name, spec in config.sidecars.items():
        parts = [f"flavor={spec.flavor or JobHardware.CPU_BASIC.value}"]
        if spec.replicas > 1:
            parts.append(f"replicas={spec.replicas}")
        if spec.healthcheck is not None:
            parts.append(f"health={spec.healthcheck.format_short()}")
        if spec.secrets:
            redacted = ", ".join(f"{k}={'***' if v is not None else '<not set>'}" for k, v in spec.secrets.items())
            parts.append(f"secrets=[{redacted}]")
        if spec.on_exit != "fail":
            parts.append(f"on_exit={spec.on_exit}")
        lines.append(f"{name} ({', '.join(parts)})")
    return "\n".join(lines)


def parse_main_job_marker(log_line: str) -> tuple[str, str | None] | None:
    """Extract `(main_job_id, main_url)` from an orchestrator log line, if present."""
    if not log_line.startswith(MAIN_JOB_ID_PREFIX):
        return None
    match = _MAIN_JOB_MARKER_RE.fullmatch(log_line.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _encode_orchestrator_source() -> str:
    from huggingface_hub import _jobs_sidecar_orchestrator

    source = inspect.getsource(_jobs_sidecar_orchestrator)
    compressed = zlib.compress(source.encode("utf-8"), level=9)
    return base64.b64encode(compressed).decode("ascii")


def build_orchestrator_job_spec(
    *,
    sidecars_config: SidecarsConfig,
    main_job_spec: dict[str, Any],
    endpoint: str,
    namespace: str,
    orchestrator_token: str,
    network_group: str | None = None,
    resource_group_id: str | None = None,
    poll_interval_secs: float = 3.0,
) -> tuple[str, dict[str, Any]]:
    """Build the orchestrator Job spec that manages the full sidecar + main Job lifecycle.

    Returns `(group_id, orchestrator_job_spec)`. All child secrets and the orchestrator API token
    are placed strictly in the orchestrator Job's `secrets` field (never in `environment`).
    """
    if network_group is not None:
        _validate_network_group(network_group)
        group_id = network_group
    else:
        group_id = f"sc-{uuid.uuid4().hex[:10]}"

    instances = sidecars_config.expand_instances()
    main_timeout_secs = int(main_job_spec.get("timeoutSeconds") or DEFAULT_JOB_TIMEOUT_SECS)
    max_startup = max(inst.spec.startup_timeout_secs for inst in instances)
    max_readiness = max(inst.spec.readiness_timeout_secs for inst in instances)
    safety_floor_secs = max_startup + max_readiness + main_timeout_secs + SAFETY_TIMEOUT_BUFFER_SECS

    # Strip secrets from main spec before placing in the non-secret manifest.
    main_spec_clean = dict(main_job_spec)
    main_secrets = main_spec_clean.pop("secrets", None)
    main_labels = dict(main_spec_clean.get("labels") or {})
    main_labels[SIDECAR_GROUP_LABEL] = group_id
    main_labels[SIDECAR_ROLE_LABEL] = "main"
    main_spec_clean["labels"] = main_labels
    main_network = dict(main_spec_clean.get("network") or {})
    main_network["group"] = group_id
    main_spec_clean["network"] = main_network
    if resource_group_id:
        main_spec_clean["resourceGroupId"] = resource_group_id

    child_secrets: dict[str, Any] = {}
    if main_secrets:
        child_secrets["main"] = main_secrets

    sidecar_entries: list[dict[str, Any]] = []
    sidecar_secrets_map: dict[str, dict[str, Any]] = {}
    max_sidecar_timeout = safety_floor_secs

    for inst in instances:
        sc = inst.spec
        sc_timeout = max(sc.timeout_secs or 0, safety_floor_secs)
        max_sidecar_timeout = max(max_sidecar_timeout, sc_timeout)

        sc_labels = {
            **sc.labels,
            SIDECAR_GROUP_LABEL: group_id,
            SIDECAR_ROLE_LABEL: "sidecar",
            SIDECAR_NAME_LABEL: inst.instance_name,
        }
        sc_job_name = sc_labels.pop("name", None) or _sanitize_job_name(
            f"sidecar-{inst.instance_name}-{group_id[-6:]}"
        )
        sc_job_spec = _create_job_spec(
            image=sc.image,
            command=sc.command,
            env=sc.env,
            secrets=None,  # Passed via encrypted orchestrator secrets envelope below.
            flavor=sc.flavor,
            timeout=sc_timeout,
            name=sc_job_name,
            labels=sc_labels,
            volumes=sc.volumes or None,
            expose=sc.expose,
            expose_public=sc.expose_public,
            ssh=sc.ssh,
            network_group=group_id,
            network_aliases=inst.network_aliases,
            resource_group_id=resource_group_id,
        )
        if sc.secrets:
            sidecar_secrets_map[inst.instance_name] = {k: v for k, v in sc.secrets.items() if v is not None}

        sidecar_entries.append(
            {
                "name": inst.instance_name,
                "service_name": inst.service_name,
                "probe_alias": inst.probe_alias,
                "startup_timeout_secs": sc.startup_timeout_secs,
                "readiness_timeout_secs": sc.readiness_timeout_secs,
                "on_exit": sc.on_exit,
                "healthcheck": sc.healthcheck.to_manifest_dict() if sc.healthcheck else None,
                "spec": sc_job_spec,
            }
        )

    if sidecar_secrets_map:
        child_secrets["sidecars"] = sidecar_secrets_map

    manifest = {
        "endpoint": endpoint,
        "namespace": namespace,
        "group_id": group_id,
        "group_label_key": SIDECAR_GROUP_LABEL,
        "poll_interval_secs": poll_interval_secs,
        "sidecars": sidecar_entries,
        "main": {"spec": main_spec_clean},
    }
    encoded_manifest = base64.b64encode(zlib.compress(json.dumps(manifest).encode("utf-8"), level=9)).decode("ascii")
    encoded_script = _encode_orchestrator_source()

    orch_secrets: dict[str, str] = {"HF_ORCHESTRATOR_TOKEN": orchestrator_token}
    if child_secrets:
        orch_secrets["HF_ORCHESTRATOR_SECRETS"] = json.dumps(child_secrets)

    orch_timeout = max_sidecar_timeout + SAFETY_TIMEOUT_BUFFER_SECS
    orch_name = _sanitize_job_name(f"sidecars-orchestrator-{group_id[-6:]}")
    orch_spec = _create_job_spec(
        image=ORCHESTRATOR_IMAGE,
        command=[
            "python3",
            "-c",
            "import base64,os,zlib; exec(compile(zlib.decompress(base64.b64decode(os.environ['HF_ORCHESTRATOR_SCRIPT'])).decode('utf-8'), '<sidecar-orchestrator>', 'exec'), {'__name__': '__main__'})",
        ],
        env={
            "HF_ORCHESTRATOR_MANIFEST": encoded_manifest,
            "HF_ORCHESTRATOR_SCRIPT": encoded_script,
        },
        secrets=orch_secrets,
        flavor=JobHardware.CPU_BASIC,
        timeout=orch_timeout,
        name=orch_name,
        labels={
            SIDECAR_GROUP_LABEL: group_id,
            SIDECAR_ROLE_LABEL: "orchestrator",
        },
        network_group=group_id,
        resource_group_id=resource_group_id,
    )
    return group_id, orch_spec
