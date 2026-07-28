# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Contains commands to interact with jobs on the Hugging Face Hub."""

import itertools
import multiprocessing
import multiprocessing.pool
import shlex
import shutil
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from queue import Empty, Queue
from typing import Annotated, Any, TypeVar
from urllib.parse import urlsplit

from huggingface_hub import HfApi, JobHardware, JobInfo, JobStage, Volume, constants
from huggingface_hub._jobs_api import (
    DEFAULT_UV_IMAGE,
    TERMINAL_JOB_STAGES,
    _default_job_name_from_image,
    _default_job_name_from_script,
)
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import logging
from huggingface_hub.utils._cache_manager import _format_size
from huggingface_hub.utils._hf_uris import _split_mount
from huggingface_hub.utils._parsing import format_duration, parse_duration

from ._cli_utils import (
    EnvFileOpt,
    EnvOpt,
    SecretsFileOpt,
    SecretsOpt,
    SoftChoice,
    SshDryRunOpt,
    SshIdentityFileOpt,
    TokenOpt,
    _get_extended_environ,
    exec_ssh,
    get_hf_api,
    parse_env_map,
    parse_volumes,
    typer_factory,
)
from ._framework import Argument, Option
from ._output import _dataclass_to_dict, out
from ._uv_script_header import TABLE_NAME, UvScript, UvScriptHeader, load_uv_script


logger = logging.get_logger(__name__)


def _parse_namespace_from_job_id(job_id: str, namespace: str | None) -> tuple[str, str | None]:
    """Extract namespace from job_id if provided in 'namespace/job_id' format.

    Allows users to pass job IDs copied from the Hub UI (e.g. 'username/job_id')
    instead of only bare job IDs. If the namespace is also provided explicitly via
    --namespace and conflicts, a CLIError is raised.
    """
    if not job_id:
        raise CLIError("Job ID cannot be empty.")

    if job_id.count("/") > 1:
        raise CLIError(f"Job ID must be in the form 'job_id' or 'namespace/job_id': '{job_id}'.")

    if "/" not in job_id:
        return job_id, namespace

    extracted_namespace, parsed_job_id = job_id.split("/", 1)
    if not extracted_namespace or not parsed_job_id:
        raise CLIError(f"Job ID must be in the form 'job_id' or 'namespace/job_id': '{job_id}'.")

    if namespace is not None and namespace != extracted_namespace:
        raise CLIError(
            f"Conflicting namespace: got --namespace='{namespace}' but job ID implies namespace='{extracted_namespace}'"
        )

    return parsed_job_id, extracted_namespace


def _parse_and_sync_job_volumes(
    volumes: list[str] | None, *, api: HfApi, namespace: str | None
) -> list[Volume] | None:
    """Parse `-v` specs for Jobs commands.

    Same as [`parse_volumes`] but the source side can also be a local directory: it is synced to a
    bucket via [`HfApi.sync_job_volume`] and the resulting bucket subfolder is mounted (read-only
    unless ':rw' is specified).
    """
    if not volumes:
        return None

    result: list[Volume] = []
    for raw_spec in volumes:
        if raw_spec.startswith(constants.HF_PROTOCOL):
            result.extend(parse_volumes([raw_spec]) or [])
            continue

        # Not a 'hf://' URI: treat the source as a local directory.
        source, mount_path, read_only = _split_mount(raw_spec, raw=raw_spec)
        if mount_path is None:
            raise CLIError(
                f"Missing mount path in volume spec '{raw_spec}'. Expected 'LOCAL_DIR:/MOUNT_PATH[:ro|:rw]' (e.g. './data:/data')."
            )
        if not Path(source).expanduser().is_dir():
            raise CLIError(
                f"Volume source '{source}' is not an existing local directory. "
                "To mount a repo or bucket instead, use the 'hf://' syntax (e.g. 'hf://buckets/my-org/my-bucket:/data')."
            )
        volume = api.sync_job_volume(
            source,
            mount_path,
            read_only=read_only if read_only is not None else True,
            namespace=namespace,
        )
        if volume.read_only is False:
            out.hint(
                f"Volume '{mount_path}' is mounted read-write. Once the job is over, pull back its data with:\n"
                f"  hf buckets sync hf://buckets/{volume.source}/{volume.path} {source}"
            )
        result.append(volume)
    return result


@dataclass
class _UvJobConfig:
    """Resolved launch configuration of a UV Job: CLI flags merged with the script's `[tool.hf-jobs]` table."""

    script: str
    """What is passed to the Jobs API. Same as the CLI argument, except for a URL script (downloaded locally)."""

    script_args: list[str] = field(default_factory=list)
    image: str | None = None
    flavor: str | None = None
    python: str | None = None
    timeout: str | None = None
    namespace: str | None = None
    env: dict[str, str | None] = field(default_factory=dict)
    secrets: dict[str, str | None] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    volume_specs: list[str] = field(default_factory=list)
    volumes: list[Volume] | None = None

    from_script: set[str] = field(default_factory=set)
    """Keys whose value comes from the script's `[tool.hf-jobs]` table, for display purposes."""

    source: UvScript | None = None
    """Kept around so that a downloaded script is not cleaned up before the Job is submitted."""


def _resolve_uv_job_config(
    *,
    api: HfApi,
    script: str,
    script_args: list[str] | None,
    dependencies: list[str] | None,
    image: str | None,
    flavor: str | None,
    python: str | None,
    env: list[str] | None,
    env_file: str | None,
    secrets: list[str] | None,
    secrets_file: str | None,
    timeout: str | None,
    name: str | None,
    label: list[str] | None,
    volume: list[str] | None,
    namespace: str | None,
    dry_run: bool,
) -> _UvJobConfig:
    """Merge the CLI flags with the `[tool.hf-jobs]` table the script carries, if any.

    An explicit CLI flag always wins over the script; `env`, `labels`, `secrets` and `volumes` are merged
    entry by entry (by key, by name and by mount path respectively), so a script value and a CLI value
    only ever conflict when they target the same entry.
    """
    source = load_uv_script(script)
    header = source.header or UvScriptHeader()
    from_script: set[str] = set()

    def pick(key: str, cli_value: str | None) -> str | None:
        """Keep the CLI value, and fall back to the script's."""
        if cli_value is not None:
            return cli_value
        if (script_value := getattr(header, key)) is not None:
            from_script.add(key)
            return script_value
        return None

    image = pick("image", image)
    flavor = pick("flavor", flavor)
    python = pick("python", python)
    timeout = pick("timeout", timeout)
    namespace = pick("namespace", namespace)

    env_map = parse_env_map(env, env_file)
    script_env = {key: value for key, value in header.env.items() if key not in env_map}
    from_script.update(f"env.{key}" for key in script_env)
    env_map = {**script_env, **env_map}

    secrets_map = parse_env_map(secrets, secrets_file)
    script_secrets = _resolve_script_secrets(header.secrets, secrets_map)
    from_script.update(f"secrets.{key}" for key in script_secrets)
    secrets_map = {**script_secrets, **secrets_map}

    labels_map = _parse_labels_map(label, name=name) or {}
    # `name = ...` in the script's table is shorthand for the `name` label.
    script_labels = {**header.labels, **({"name": header.name} if header.name is not None else {})}
    script_labels = {key: value for key, value in script_labels.items() if key not in labels_map}
    from_script.update(f"labels.{key}" for key in script_labels)
    labels_map = {**script_labels, **labels_map}

    volume_specs, script_volume_specs = _merge_volume_specs(volume or [], header.volumes)
    from_script.update(f"volumes.{spec}" for spec in script_volume_specs)

    config = _UvJobConfig(
        script=source.script,
        script_args=script_args or [],
        image=image,
        flavor=flavor,
        python=python,
        timeout=timeout,
        namespace=namespace,
        env=env_map,
        secrets=secrets_map,
        labels=labels_map,
        volume_specs=volume_specs,
        # A dry run must not have side effects: local directories are only synced to a bucket for real runs.
        volumes=None if dry_run else _parse_and_sync_job_volumes(volume_specs, api=api, namespace=namespace),
        from_script=from_script,
        source=source,
    )
    if "name" not in config.labels:
        # `script` (not `config.script`) so that a URL keeps naming the Job after the URL, not after
        # the temporary file it was downloaded to.
        config.labels["name"] = _default_job_name_from_script(
            script,
            config.script_args,
            config_parts=_name_hash_parts(
                flavor=config.flavor,
                timeout=config.timeout,
                namespace=config.namespace,
                env=config.env,
                secrets=config.secrets,
                volume_specs=config.volume_specs,
                extra=[config.image or DEFAULT_UV_IMAGE, config.python or "", *(dependencies or [])],
            ),
        )
    return config


def _resolve_script_secrets(names: list[str], cli_secrets: dict[str, str | None]) -> dict[str, str]:
    """Resolve the secrets requested by a script from the caller's environment.

    A script only lists secret *names*: values always come from whoever runs it (`HF_TOKEN` also
    resolves from `hf auth login`). A requested secret that is not set locally is an error rather than
    an empty value: forgetting to export a secret before launching is easy, and debugging the Job that
    results from it is not.
    """
    environ = _get_extended_environ()
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        if name in cli_secrets:  # an explicit `--secrets NAME=...` wins
            continue
        if name in environ:
            resolved[name] = environ[name]
        else:
            missing.append(name)
    if missing:
        raise CLIError(
            f"The script requires the following secret(s), which are not set in your environment: {', '.join(missing)}."
            f" Export them locally (e.g. `export {missing[0]}=...`) or pass them explicitly"
            f" (e.g. `--secrets {missing[0]}=...`)."
        )
    if resolved:
        out.warning(
            f"The script's [{TABLE_NAME}] table requests {', '.join(resolved)}:"
            " the value(s) from your local environment will be sent to the Job."
        )
    return resolved


def _merge_volume_specs(cli_specs: list[str], script_specs: list[str]) -> tuple[list[str], list[str]]:
    """Merge `-v` specs with the ones from the script. Returns `(all specs, specs from the script)`.

    A CLI mount overrides the script's mount at the same path.
    """
    cli_mount_paths = {_split_mount(spec, raw=spec)[1] for spec in cli_specs}
    kept = [spec for spec in script_specs if _split_mount(spec, raw=spec)[1] not in cli_mount_paths]
    return kept + cli_specs, kept


def _name_hash_parts(
    *,
    flavor: str | None,
    timeout: str | None,
    namespace: str | None,
    env: dict[str, str | None],
    secrets: dict[str, str | None],
    volume_specs: list[str],
    extra: list[str] | None = None,
) -> list[str]:
    """The resolved launch values hashed into a Job's default name, on top of its image/script.

    Same inputs for `hf jobs run` and `hf jobs uv run`, so that both name Jobs after what they actually
    submit (`extra` carries what is specific to UV runs). Env *values* are part of the hash since they
    change what the Job does; secrets only contribute their names, never their values.
    """
    return [
        flavor or JobHardware.CPU_BASIC.value,
        timeout or "",
        namespace or "",
        *(extra or []),
        *(f"{key}={value}" for key, value in sorted(env.items())),
        *sorted(secrets),
        *volume_specs,
    ]


_FROM_SCRIPT = " (from script)"


def _print_job_summary(values: dict[str, Any], *, from_script: Collection[str] = (), dry_run: bool = False) -> None:
    """Echo the launch configuration of a Job before submitting it.

    Printed on every run, so that what is sent to the Jobs API is always visible - in particular the
    values injected by a script's `[tool.hf-jobs]` table, which are marked as such. Goes to stderr for a
    real run, and to stdout for `--dry-run`, where it is the output of the command.

    Multi-entry values (env, secrets, ...) are pre-formatted by the `_format_*` helpers, one entry per
    line, since each entry can come from the script or from the command line.
    """
    rows = [(key, str(value)) for key, value in values.items() if value not in (None, "", [], {}, False)]
    width = max(len(key) for key, _ in rows)
    lines = ["Job configuration:"]
    for key, value in rows:
        value += _FROM_SCRIPT if key in from_script else ""
        # Only the first line of a multi-line value is prefixed with the key.
        lines += [
            f"  {(key if index == 0 else '').ljust(width)}  {line}" for index, line in enumerate(value.split("\n"))
        ]
    summary = "\n".join(lines)
    if dry_run:
        out.text(f"{summary}\n(dry run) Job not submitted.")
    else:
        out.log(summary)


# Env values can be arbitrarily long (a JSON blob, a prompt, ...): keep the summary readable.
_MAX_DISPLAYED_ENV_VALUE = 60


def _format_env(env: dict[str, str | None], *, from_script: Collection[str] = ()) -> str:
    return _format_entries({key: f"{key}={_ellipsis(value or '')}" for key, value in env.items()}, "env", from_script)


def _format_secrets(secrets: dict[str, str | None], *, from_script: Collection[str] = ()) -> str:
    """Format secrets for display: names only, values always redacted."""
    return _format_entries({key: f"{key}=***" for key in secrets}, "secrets", from_script)


def _format_labels(labels: dict[str, str], *, from_script: Collection[str] = ()) -> str:
    return _format_entries({key: f"{key}={value}" for key, value in labels.items()}, "labels", from_script)


def _format_volumes(volume_specs: list[str], *, from_script: Collection[str] = ()) -> str:
    return _format_entries({spec: spec for spec in volume_specs}, "volumes", from_script)


def _format_entries(entries: dict[str, str], key: str, from_script: Collection[str]) -> str:
    """Render one entry per line, marking the entries that come from the script's `[tool.hf-jobs]` table."""
    return "\n".join(text + (_FROM_SCRIPT if f"{key}.{name}" in from_script else "") for name, text in entries.items())


def _ellipsis(value: str) -> str:
    return value if len(value) <= _MAX_DISPLAYED_ENV_VALUE else value[: _MAX_DISPLAYED_ENV_VALUE - 1] + "…"


STATS_UPDATE_MIN_INTERVAL = 0.1  # we set a limit here since there is one update per second per job

# Common job-related options
ImageArg = Annotated[
    str,
    Argument(
        help="The Docker image to use.",
    ),
]

ImageOpt = Annotated[
    str | None,
    Option(
        help="Use a custom Docker image with `uv` installed.",
    ),
]

FlavorOpt = Annotated[
    str | None,
    Option(
        help="Flavor for the hardware. Run 'hf jobs hardware' to list available flavors. Defaults to `cpu-basic`.",
        click_type=SoftChoice(JobHardware),
    ),
]

LabelsOpt = Annotated[
    list[str] | None,
    Option(
        "-l",
        "--label",
        help="Set labels. E.g. --label KEY=VALUE or --label LABEL",
    ),
]

NameOpt = Annotated[
    str | None,
    Option(
        "--name",
        help="Name the Job. Stored as the `name` label. Names do not have to be unique. Defaults to the image or script name plus a short hash of the command.",
    ),
]

TimeoutOpt = Annotated[
    str | None,
    Option(
        help="Max duration: int with s (seconds, default), m (minutes), h (hours) or d (days).",
    ),
]

DetachOpt = Annotated[
    bool,
    Option(
        "-d",
        "--detach",
        help="Run the Job in the background and print the Job ID.",
    ),
]

DryRunOpt = Annotated[
    bool,
    Option(
        "--dry-run",
        help="Print the resolved Job configuration without submitting the Job.",
    ),
]

NamespaceOpt = Annotated[
    str | None,
    Option(
        help="The namespace where the job will be running. Defaults to the current user's namespace.",
    ),
]

ResourceGroupIdOpt = Annotated[
    str | None,
    Option(
        "--resource-group-id",
        help="The ID of the resource group to create the Job in. Used to control access to resources within an organization.",
    ),
]

ExposeOpt = Annotated[
    list[int] | None,
    Option(
        "--expose",
        help="Expose a container port through the jobs proxy. Repeat the flag for multiple ports (e.g. `--expose 8000 --expose 8001`). Each exposed port is reachable on the public jobs domain; access requires an HF token with read access to the job's namespace.",
    ),
]

SshEnabledOpt = Annotated[
    bool,
    Option(
        "--ssh",
        help="Make the job's container reachable over SSH. Connect with `hf jobs ssh <job_id>`. Requires an SSH public key registered on https://huggingface.co/settings/keys.",
    ),
]

WithOpt = Annotated[
    list[str] | None,
    Option(
        "--with",
        help="Run with the given packages installed",
    ),
]

PythonOpt = Annotated[
    str | None,
    Option(
        "-p",
        "--python",
        help="The Python interpreter to use for the run environment",
    ),
]

SuspendOpt = Annotated[
    bool | None,
    Option(
        help="Suspend (pause) the scheduled Job",
    ),
]

ConcurrencyOpt = Annotated[
    bool | None,
    Option(
        help="Allow multiple instances of this Job to run concurrently",
    ),
]

ScheduleArg = Annotated[
    str,
    Argument(
        help="One of annually, yearly, monthly, weekly, daily, hourly, or a CRON schedule expression.",
    ),
]

ScriptArg = Annotated[
    str,
    Argument(
        help="UV script to run (local file or URL)",
    ),
]

ScriptArgsArg = Annotated[
    list[str] | None,
    Argument(
        help="Arguments for the script",
    ),
]


CommandArg = Annotated[
    list[str],
    Argument(
        help="The command to run.",
    ),
]

JobIdArg = Annotated[
    str,
    Argument(
        help="Job ID (or 'namespace/job_id')",
    ),
]

JobIdsArg = Annotated[
    list[str] | None,
    Argument(
        help="Job IDs (or 'namespace/job_id')",
    ),
]

ScheduledJobIdArg = Annotated[
    str,
    Argument(
        help="Scheduled Job ID (or 'namespace/scheduled_job_id')",
    ),
]

JobVolumesOpt = Annotated[
    list[str] | None,
    Option(
        "-v",
        "--volume",
        help="Mount one or more volumes. Format: hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro|:rw] or LOCAL_DIR:/MOUNT_PATH[:ro|:rw]. "
        "TYPE is one of: models, datasets, spaces, buckets. "
        "TYPE defaults to models if omitted. "
        "models, datasets and spaces are always mounted read-only. buckets are read+write by default. "
        "A local directory source is first synced to a bucket and mounted read-only by default. "
        "E.g. -v hf://datasets/org/ds:/data or -v hf://buckets/org/b:/mnt:ro or -v ./inputs:/inputs",
    ),
]


jobs_cli = typer_factory(help="Run and manage Jobs on the Hub.")


def _stream_logs_and_check_status(api: HfApi, job: JobInfo) -> None:
    """Stream Job logs until the Job ends, then fail the command if the Job did not complete successfully."""
    for log in api.fetch_job_logs(job_id=job.id, namespace=job.owner.name, follow=True):
        out.text(log)
    # The log stream can end while the Job is still scheduling or shutting down: settle the final state.
    final = api.wait_for_job(job_id=job.id, namespace=job.owner.name)
    if final.status.stage != JobStage.COMPLETED:
        message = f": {final.status.message}" if final.status.message else ""
        raise CLIError(f"Job {final.id} finished with stage '{final.status.stage}'{message}")
    out.text(f"Job {final.id} completed")


@jobs_cli.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "hf jobs run --name hello-world python:3.12 python -c 'print(\"Hello!\")'",
        "hf jobs run --detach python:3.12 python script.py",
        "hf jobs run -e FOO=foo python:3.12 python script.py",
        "hf jobs run --secrets HF_TOKEN python:3.12 python script.py",
        "hf jobs run -v hf://org/my-model:/data -v hf://buckets/org/b:/mnt python:3.12 python script.py",
    ],
)
def jobs_run(
    image: ImageArg,
    command: CommandArg,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    detach: DetachOpt = False,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    ssh: SshEnabledOpt = False,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Run a Job."""
    env_map = parse_env_map(env, env_file)
    secrets_map = parse_env_map(secrets, secrets_file)
    labels_map = _parse_labels_map(label, name=name) or {}
    labels_map.setdefault(
        "name",
        _default_job_name_from_image(
            image,
            command,
            config_parts=_name_hash_parts(
                flavor=flavor,
                timeout=timeout,
                namespace=namespace,
                env=env_map,
                secrets=secrets_map,
                volume_specs=volume or [],
            ),
        ),
    )

    api = get_hf_api(token=token)
    volumes = None if dry_run else _parse_and_sync_job_volumes(volume, api=api, namespace=namespace)
    _print_job_summary(
        {
            "image": image,
            "command": shlex.join(command),
            "flavor": flavor or JobHardware.CPU_BASIC.value,
            "timeout": timeout,
            "env": _format_env(env_map),
            "secrets": _format_secrets(secrets_map),
            "volumes": _format_volumes(volume or []),
            "labels": _format_labels(labels_map),
            "expose": " ".join(str(port) for port in expose or []),
            "ssh": ssh,
            "namespace": namespace,
        },
        dry_run=dry_run,
    )
    if dry_run:
        return
    job = api.run_job(
        image=image,
        command=command,
        env=env_map,
        secrets=secrets_map,
        labels=labels_map,
        volumes=volumes,
        flavor=flavor,
        timeout=timeout,
        expose=expose,
        ssh=ssh,
        resource_group_id=resource_group_id,
        namespace=namespace,
    )
    out.result("Job started", id=job.id, name=(job.labels or {}).get("name"), url=job.url)
    if not _has_explicit_name(name, label):
        auto_name = (job.labels or {}).get("name")
        out.hint(
            f"Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs labels {job.owner.name}/{job.id} --name NAME` to rename it."
        )
    if isinstance(job.status.expose_urls, list):
        urls = "\n".join(f"  {url}" for url in job.status.expose_urls)
        out.hint(f"Exposed ports are reachable at (requires an HF token with read access to the job):\n{urls}")
    if isinstance(job.status.ssh_url, str):
        out.hint(f"Use `hf jobs ssh {job.owner.name}/{job.id}` to open an SSH session into the job.")
    if detach:
        job_ref = f"{job.owner.name}/{job.id}"
        out.hint(f"Use `hf jobs logs -f {job_ref}` to stream logs, or `hf jobs inspect {job_ref}` to check status.")
        out.hint(f"Use `hf jobs wait {job_ref}` to block until it finishes.")
        return
    _stream_logs_and_check_status(api, job)


@jobs_cli.command(
    "logs",
    examples=[
        "hf jobs logs <job_id>",
        "hf jobs logs -f <job_id>",
        "hf jobs logs --tail 20 <job_id>",
        "hf jobs logs -f --tail 100 <job_id>",
    ],
)
def jobs_logs(
    job_id: JobIdArg,
    follow: Annotated[
        bool,
        Option(
            "-f",
            "--follow",
            help="Follow log output (stream until the job completes). Without this flag, only currently available logs are printed.",
        ),
    ] = False,
    tail: Annotated[
        int | None,
        Option(
            "-n",
            "--tail",
            help="Number of lines to show from the end of the logs. When combined with --follow, starts streaming from the last N lines.",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Fetch the logs of a Job.

    By default, prints currently available logs and exits (non-blocking).
    Use --follow/-f to stream logs in real-time until the job completes.
    Use --tail/-n to limit the number of lines returned (server-side when supported).

    Note: following exits when the log stream ends, regardless of whether the Job
    succeeded or failed. Run `hf jobs inspect <job_id>` to check the final status.
    """
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)

    api = get_hf_api(token=token)
    logs = api.fetch_job_logs(job_id=job_id, namespace=namespace, follow=follow, tail=tail)
    for log in logs:
        out.text(log)
    if follow:
        job_ref = f"{namespace}/{job_id}" if namespace else job_id
        out.hint(f"Stream ended. Run `hf jobs inspect {job_ref}` to check the final status (e.g. COMPLETED or ERROR).")


def _matches_filters(job_properties: dict[str, str], filters: list[tuple[str, str, str]]) -> bool:
    """Check if scheduled job matches all specified filters."""
    for key, op_str, pattern in filters:
        value = job_properties.get(key)
        if value is None:
            if op_str == "!=":
                continue
            return False
        match = fnmatch(value.lower(), pattern.lower())
        if (op_str == "=" and not match) or (op_str == "!=" and match):
            return False
    return True


def _clear_line(n: int) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def _get_jobs_stats_rows(
    job_id: str, metrics_stream: Iterable[dict[str, Any]], table_headers: list[str]
) -> Iterable[tuple[bool, str, list[list[str | int]]]]:
    for metrics in metrics_stream:
        row = [
            job_id,
            f"{metrics['cpu_usage_pct']}%",
            round(metrics["cpu_millicores"] / 1000.0, 1),
            f"{round(100 * metrics['memory_used_bytes'] / metrics['memory_total_bytes'], 2)}%",
            f"{_format_size(metrics['memory_used_bytes'])}B / {_format_size(metrics['memory_total_bytes'])}B",
            f"{_format_size(metrics['rx_bps'])}bps / {_format_size(metrics['tx_bps'])}bps",
        ]
        if metrics["gpus"] and isinstance(metrics["gpus"], dict):
            rows = [row] + [[""] * len(row)] * (len(metrics["gpus"]) - 1)
            for row, gpu_id in zip(rows, sorted(metrics["gpus"])):
                gpu = metrics["gpus"][gpu_id]
                row += [
                    f"{gpu['utilization']}%",
                    f"{round(100 * gpu['memory_used_bytes'] / gpu['memory_total_bytes'], 2)}%",
                    f"{_format_size(gpu['memory_used_bytes'])}B / {_format_size(gpu['memory_total_bytes'])}B",
                ]
        else:
            row += ["N/A"] * (len(table_headers) - len(row))
            rows = [row]
        yield False, job_id, rows
    yield True, job_id, []


@jobs_cli.command("stats", examples=["hf jobs stats <job_id>"])
def jobs_stats(
    job_ids: JobIdsArg = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Fetch the resource usage statistics and metrics of Jobs"""
    if job_ids is not None:
        parsed_ids = []
        for job_id in job_ids:
            job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
            parsed_ids.append(job_id)
        job_ids = parsed_ids
    api = get_hf_api(token=token)
    if namespace is None:
        namespace = api.whoami()["name"]
    if job_ids is None:
        job_ids = [
            job.id
            for job in api.list_jobs(namespace=namespace)
            if (job.status.stage if job.status else "UNKNOWN") in ("RUNNING", "UPDATING")
        ]
    if len(job_ids) == 0:
        out.text("No running jobs found")
        return
    table_headers = [
        "JOB ID",
        "CPU %",
        "NUM CPU",
        "MEM %",
        "MEM USAGE",
        "NET I/O",
        "GPU UTIL %",
        "GPU MEM %",
        "GPU MEM USAGE",
    ]
    with multiprocessing.pool.ThreadPool(len(job_ids)) as pool:
        rows_per_job_id: dict[str, list[list[str | int]]] = {}
        for job_id in job_ids:
            row: list[str | int] = [job_id]
            row += ["-- / --" if ("/" in header or "USAGE" in header) else "--" for header in table_headers[1:]]
            rows_per_job_id[job_id] = [row]
        last_update_time = time.time()
        total_rows = [row for job_id in rows_per_job_id for row in rows_per_job_id[job_id]]
        # In-place refresh (cursor-up + clear) requires a fixed line count and layout —
        # `out.table`'s mode-dependent formatting would break it.
        print(_tabulate(total_rows, headers=table_headers))

        kwargs_list = [
            {
                "job_id": job_id,
                "metrics_stream": api.fetch_job_metrics(job_id=job_id, namespace=namespace),
                "table_headers": table_headers,
            }
            for job_id in job_ids
        ]
        for done, job_id, rows in iflatmap_unordered(pool, _get_jobs_stats_rows, kwargs_list=kwargs_list):
            if done:
                rows_per_job_id.pop(job_id, None)
            else:
                rows_per_job_id[job_id] = rows
            now = time.time()
            if now - last_update_time >= STATS_UPDATE_MIN_INTERVAL:
                _clear_line(2 + len(total_rows))
                total_rows = [row for job_id in rows_per_job_id for row in rows_per_job_id[job_id]]
                print(_tabulate(total_rows, headers=table_headers))
                last_update_time = now


@jobs_cli.command(
    "list | ls | ps",
    examples=[
        "hf jobs ls",
        "hf jobs ls -a",
        "hf jobs ls --status running,scheduling",
        "hf jobs ls --name training-v2",
        "hf jobs ls --label env=prod --label team=ml",
        "hf jobs ls --all --label hf-sandbox=1",
    ],
)
def jobs_ps(
    all: Annotated[
        bool,
        Option(
            "-a",
            "--all",
            help="Show all Jobs (default shows running and scheduling). Cannot be combined with --status.",
        ),
    ] = False,
    status: Annotated[
        list[str] | None,
        Option(
            "--status",
            click_type=SoftChoice(JobStage),
            help="Only show Jobs with the given status. Comma-separated or repeated, e.g. `--status running,scheduling`.",
        ),
    ] = None,
    label: Annotated[
        list[str] | None,
        Option(
            "-l",
            "--label",
            help="Only show Jobs with the given `key=value` label. Repeat to require several labels, e.g. `--label env=prod --label team=ml`.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        Option(
            "--name",
            help="Only show Jobs with the given name (shortcut for `--label name=NAME`).",
        ),
    ] = None,
    limit: Annotated[
        int,
        Option(
            "--limit",
            help="Maximum number of Jobs to display. Set to 0 to show all (no limit).",
        ),
    ] = 100,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    filter: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter",
            help="(Deprecated) Use `--status` and `--label` instead.",
        ),
    ] = None,
) -> None:
    """List Jobs.

    Use `--status` to filter by status (see [`JobStage`] for possible values) and `--label` to filter by `key=value`
    labels. A Job must match every filter to be listed.
    """
    api = get_hf_api(token=token)

    if filter:
        out.warning(
            f"Ignoring filter '{filter}'."
            " `-f`/`--filter` is deprecated and will be removed in a future release. Use `--status`/`--label`."
        )

    if all and status:
        raise CLIError("`-a`/`--all` cannot be combined with `--status`.")

    # Status filtering (default to active Jobs, unless `--all` or `--status` is provided).
    raw_statuses: list[str] = []
    for value in status or []:
        raw_statuses.extend(part.strip() for part in value.split(",") if part.strip())

    server_statuses: list[str] | None
    if raw_statuses:
        server_statuses = raw_statuses
    elif all:
        server_statuses = None
    else:
        server_statuses = [JobStage.RUNNING.value, JobStage.SCHEDULING.value]

    # Labels filtering
    labels: dict[str, str] = {}
    for item in label or []:
        if "=" not in item:
            raise CLIError(f"Invalid label filter '{item}': must be in the form 'key=value'")
        key, value = item.split("=")
        labels[key] = value

    # `--name` is a shortcut for the `name` label.
    if name is not None:
        if "name" in labels:
            raise CLIError("Cannot filter by both `--name` and `--label name=...`.")
        labels["name"] = name

    jobs_iter = api.list_jobs(namespace=namespace, status=server_statuses, labels=labels or None)

    # Apply the display limit. Fetch one extra Job to detect (and warn about) truncation.
    truncated = False
    if limit > 0:
        jobs = list(itertools.islice(jobs_iter, limit + 1))
        if len(jobs) > limit:
            truncated = True
            jobs = jobs[:limit]
    else:
        jobs = list(jobs_iter)

    # Build display items. Augment the raw api dict with curated, table-friendly columns.
    job_items: list[dict[str, Any]] = []
    for job in jobs:
        job_item = _dataclass_to_dict(job)
        durations = job_item.get("durations") or {}
        cmd = job_item.get("command") or []
        job_item["job_id"] = job_item.get("id", "")
        job_item["name"] = (job_item.get("labels") or {}).get("name") or "N/A"
        job_item["image/space"] = job_item.get("docker_image") or "N/A"
        job_item["command"] = " ".join(cmd) if cmd else "N/A"
        job_item["created"] = job_item["created_at"][:19].replace("T", " ") if job_item.get("created_at") else "N/A"
        job_item["status"] = (job_item.get("status") or {}).get("stage", "UNKNOWN")
        job_item["runtime"] = format_duration(durations.get("running_secs"))
        job_items.append(job_item)

    out.table(
        job_items,
        headers=["job_id", "name", "image/space", "command", "created", "status", "runtime"],
        id_key="job_id",
    )
    if truncated:
        out.hint(f"Output truncated to {limit} Jobs. Use `--limit 0` to show all (or `--limit N`).")
    if not job_items:
        if raw_statuses or labels:
            filters_msg = ", ".join(
                [*(f"status={s}" for s in raw_statuses), *(f"label={k}={v}" for k, v in labels.items())]
            )
            out.text(f"No jobs matched filters: {filters_msg}")
        elif not all:
            out.hint("No running jobs. Use `-a`/`--all` to include finished (and failed) jobs.")


@jobs_cli.command("hardware", examples=["hf jobs hardware"])
def jobs_hardware() -> None:
    """List available hardware options for Jobs"""
    api = get_hf_api()
    hardware_list = api.list_jobs_hardware()
    items = []
    for hw in hardware_list:
        accelerator_info = ""
        if hw.accelerator:
            accelerator_info = f"{hw.accelerator.quantity}x {hw.accelerator.model} ({hw.accelerator.vram})"
        cost_min = f"${hw.unit_cost_usd:.4f}" if hw.unit_cost_usd else "free"
        cost_hour = f"${hw.unit_cost_usd * 60:.2f}" if hw.unit_cost_usd else "free"
        items.append(
            {
                "name": hw.name,
                "pretty name": hw.pretty_name,
                "cpu": hw.cpu,
                "ram": hw.ram,
                "storage": hw.ephemeral_storage,
                "accelerator": accelerator_info,
                "cost/min": cost_min,
                "cost/hour": cost_hour,
            }
        )
    out.table(items)
    out.hint("Use `hf jobs run --flavor <name> ...` to request a specific hardware flavor.")


@jobs_cli.command("inspect", examples=["hf jobs inspect <job_id>"])
def jobs_inspect(
    job_ids: Annotated[
        list[str],
        Argument(
            help="Job IDs to inspect (or 'namespace/job_id')",
        ),
    ],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Display detailed information on one or more Jobs"""
    parsed_ids = []
    for job_id in job_ids:
        job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
        parsed_ids.append(job_id)
    job_ids = parsed_ids
    api = get_hf_api(token=token)
    jobs = [api.inspect_job(job_id=job_id, namespace=namespace) for job_id in job_ids]
    out.table([_surface_name(_dataclass_to_dict(job), labels=job.labels) for job in jobs], id_key="id")


@jobs_cli.command("cancel", examples=["hf jobs cancel <job_id>"])
def jobs_cancel(
    job_id: JobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Cancel a Job"""
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    api.cancel_job(job_id=job_id, namespace=namespace)
    out.result("Job cancelled", id=job_id)


@jobs_cli.command(
    "wait",
    examples=[
        "hf jobs wait <job_id>",
        "hf jobs wait <job_id_1> <job_id_2>",
        "hf jobs ls -q | xargs hf jobs wait",
    ],
)
def jobs_wait(
    job_ids: Annotated[
        list[str],
        Argument(
            help="Job IDs to wait for (or 'namespace/job_id').",
        ),
    ],
    timeout: Annotated[
        str | None,
        Option(
            help="Max time to wait: int with s (seconds, default), m (minutes), h (hours) or d (days).",
        ),
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Wait for one or more Jobs to reach a terminal state.

    Blocks until every Job has finished, then exits with code 0 if all Jobs completed
    successfully, or a non-zero exit code if any Job was canceled, errored or deleted.

    All Jobs must belong to the same namespace.
    """
    parsed_ids = []
    namespaces = set()
    for job_id in job_ids:
        parsed_id, parsed_namespace = _parse_namespace_from_job_id(job_id, namespace)
        parsed_ids.append(parsed_id)
        namespaces.add(parsed_namespace)
    if len(namespaces) > 1:
        raise CLIError(
            "All Job IDs must be in the same namespace, got: "
            + ", ".join(str(ns) for ns in sorted(namespaces, key=str))
        )
    namespace = namespaces.pop()
    timeout_secs = parse_duration(timeout) if timeout is not None else None

    api = get_hf_api(token=token)
    status = out.status(f"Waiting for {len(parsed_ids)} Job(s) to finish...")
    try:
        jobs = api.wait_for_job(parsed_ids, timeout=timeout_secs, namespace=namespace)
    except TimeoutError:
        status.done("Timed out.")
        raise CLIError(f"Timed out after {timeout} waiting for Job(s) to finish.") from None
    status.done(f"{len(jobs)} Job(s) finished.")

    out.table([{"id": job.id, "stage": str(job.status.stage), "message": job.status.message} for job in jobs])
    failed = [job for job in jobs if job.status.stage != JobStage.COMPLETED]
    if failed:
        raise CLIError(
            f"{len(failed)} of {len(jobs)} Job(s) did not complete successfully: "
            + ", ".join(f"{job.id} ({job.status.stage})" for job in failed)
        )


@jobs_cli.command(
    "labels",
    examples=[
        "hf jobs labels <job_id> --name training-v2",
        "hf jobs labels <job_id> --label env=prod --label team=ml",
        "hf jobs labels <job_id> --clear",
    ],
)
def jobs_labels(
    job_id: JobIdArg,
    name: NameOpt = None,
    label: LabelsOpt = None,
    clear: Annotated[bool, Option("--clear", help="Remove all labels from the job.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Update labels on a Job. Passing --label replaces all existing labels; passing --name alone keeps them."""
    if not label and name is None and not clear:
        raise CLIError(
            "Please set a name with --name or at least one label with --label. To remove all labels, pass --clear."
        )
    if (label or name is not None) and clear:
        raise CLIError(
            "Cannot set a name or labels and clear them at the same time. Please use --name/--label or --clear, not both."
        )
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    if name is not None and not label:
        # Naming a Job should not wipe its existing labels: fetch them and merge the name in.
        current_labels = api.inspect_job(job_id=job_id, namespace=namespace).labels or {}
        labels = {**current_labels, "name": name}
    else:
        labels = _parse_labels_map(label, name=name) or {}
    job = api.update_job_labels(job_id=job_id, labels=labels, namespace=namespace)
    out.result("Labels updated", id=job.id, name=labels.get("name"))


@jobs_cli.command(
    "ssh",
    examples=[
        "hf jobs ssh <job_id>",
        "hf jobs ssh <job_id> --dry-run",
        "hf jobs ssh <job_id> -i ~/.ssh/id_ed25519",
    ],
)
def jobs_ssh(
    job_id: JobIdArg,
    identity_file: SshIdentityFileOpt = None,
    dry_run: SshDryRunOpt = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """SSH into a running Job.

    If the Job is not yet running, waits until it reaches the RUNNING state before
    connecting. Requires the Job to be started with SSH enabled (`hf jobs run --ssh ...`)
    and your SSH public key to be registered at https://huggingface.co/settings/keys.
    """
    job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
    api = get_hf_api(token=token)
    job = api.inspect_job(job_id=job_id, namespace=namespace)
    if job.status.ssh_url is None:
        raise CLIError("SSH is not enabled on this job. Start a job with SSH support using `hf jobs run --ssh ...`.")
    if job.status.stage in TERMINAL_JOB_STAGES:
        raise CLIError(f"Cannot SSH into job '{job.id}': job has already finished (stage: '{job.status.stage}').")
    if job.status.stage != JobStage.RUNNING:
        status = out.status(f"Waiting for job '{job.id}' to be running (stage: '{job.status.stage}')...")
        job = api.wait_for_job(job_id=job.id, namespace=namespace, stages=[JobStage.RUNNING])
        if job.status.stage != JobStage.RUNNING:
            status.done("Job finished.")
            raise CLIError(
                f"Cannot SSH into job '{job.id}': job finished before reaching RUNNING (stage: '{job.status.stage}')."
            )
        status.done("Job is running.")
    ssh_url = urlsplit(job.status.ssh_url)
    exec_ssh(
        f"{ssh_url.username}@{ssh_url.hostname}",  # type: ignore
        port=ssh_url.port,
        identity_file=identity_file,
        dry_run=dry_run,
    )


uv_app = typer_factory(help="Run UV scripts (Python with inline dependencies) on HF infrastructure.")
jobs_cli.add_group(uv_app, name="uv")


@uv_app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "hf jobs uv run --name my-script my_script.py",
        "hf jobs uv run --detach my_script.py",
        "hf jobs uv run ml_training.py --flavor a10g-small",
        "hf jobs uv run --with transformers train.py",
        "hf jobs uv run -v hf://org/my-model:/data -v hf://buckets/org/b:/mnt script.py",
        "hf jobs uv run --dry-run script.py",
    ],
)
def jobs_uv_run(
    script: ScriptArg,
    script_args: ScriptArgsArg = None,
    image: ImageOpt = None,
    flavor: FlavorOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    timeout: TimeoutOpt = None,
    detach: DetachOpt = False,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    ssh: SshEnabledOpt = False,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    with_: WithOpt = None,
    python: PythonOpt = None,
) -> None:
    """Run a UV script (local file or URL) on HF infrastructure"""
    api = get_hf_api(token=token)
    config = _resolve_uv_job_config(
        api=api,
        script=script,
        script_args=script_args,
        dependencies=with_,
        image=image,
        flavor=flavor,
        python=python,
        env=env,
        env_file=env_file,
        secrets=secrets,
        secrets_file=secrets_file,
        timeout=timeout,
        name=name,
        label=label,
        volume=volume,
        namespace=namespace,
        dry_run=dry_run,
    )
    _print_job_summary(
        {
            "script": script,
            "args": shlex.join(config.script_args),
            "with": " ".join(with_ or []),
            "image": config.image or DEFAULT_UV_IMAGE,
            "flavor": config.flavor or JobHardware.CPU_BASIC.value,
            "python": config.python,
            "timeout": config.timeout,
            "env": _format_env(config.env, from_script=config.from_script),
            "secrets": _format_secrets(config.secrets, from_script=config.from_script),
            "volumes": _format_volumes(config.volume_specs, from_script=config.from_script),
            "labels": _format_labels(config.labels, from_script=config.from_script),
            "expose": " ".join(str(port) for port in expose or []),
            "ssh": ssh,
            "namespace": config.namespace,
        },
        from_script=config.from_script,
        dry_run=dry_run,
    )
    if dry_run:
        return
    job = api.run_uv_job(
        script=config.script,
        script_args=config.script_args,
        dependencies=with_,
        python=config.python,
        image=config.image,
        env=config.env,
        secrets=config.secrets,
        labels=config.labels,
        volumes=config.volumes,
        flavor=config.flavor,
        timeout=config.timeout,
        expose=expose,
        ssh=ssh,
        resource_group_id=resource_group_id,
        namespace=config.namespace,
    )
    out.result("Job started", id=job.id, name=(job.labels or {}).get("name"), url=job.url)
    if not _has_explicit_name(name, label, config.from_script):
        auto_name = (job.labels or {}).get("name")
        out.hint(
            f"Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs labels {job.owner.name}/{job.id} --name NAME` to rename it."
        )
    if isinstance(job.status.expose_urls, list):
        urls = "\n".join(f"  {url}" for url in job.status.expose_urls)
        out.hint(f"Exposed ports are reachable at (requires an HF token with read access to the job):\n{urls}")
    if isinstance(job.status.ssh_url, str):
        out.hint(f"Use `hf jobs ssh {job.owner.name}/{job.id}` to open an SSH session into the job.")
    if detach:
        job_ref = f"{job.owner.name}/{job.id}"
        out.hint(f"Use `hf jobs logs -f {job_ref}` to stream logs, or `hf jobs inspect {job_ref}` to check status.")
        out.hint(f"Use `hf jobs wait {job_ref}` to block until it finishes.")
        return
    _stream_logs_and_check_status(api, job)


scheduled_app = typer_factory(help="Create and manage scheduled Jobs on the Hub.")
jobs_cli.add_group(scheduled_app, name="scheduled")


@scheduled_app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=['hf jobs scheduled run "0 0 * * *" --name daily-script python:3.12 python script.py'],
)
def scheduled_run(
    schedule: ScheduleArg,
    image: ImageArg,
    command: CommandArg,
    suspend: SuspendOpt = None,
    concurrency: ConcurrencyOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Schedule a Job."""
    env_map = parse_env_map(env, env_file)
    secrets_map = parse_env_map(secrets, secrets_file)
    labels_map = _parse_labels_map(label, name=name) or {}
    labels_map.setdefault(
        "name",
        _default_job_name_from_image(
            image,
            command,
            config_parts=_name_hash_parts(
                flavor=flavor,
                timeout=timeout,
                namespace=namespace,
                env=env_map,
                secrets=secrets_map,
                volume_specs=volume or [],
            ),
        ),
    )

    api = get_hf_api(token=token)
    volumes = None if dry_run else _parse_and_sync_job_volumes(volume, api=api, namespace=namespace)
    _print_job_summary(
        {
            "schedule": schedule,
            "image": image,
            "command": shlex.join(command),
            "flavor": flavor or JobHardware.CPU_BASIC.value,
            "timeout": timeout,
            "env": _format_env(env_map),
            "secrets": _format_secrets(secrets_map),
            "volumes": _format_volumes(volume or []),
            "labels": _format_labels(labels_map),
            "expose": " ".join(str(port) for port in expose or []),
            "namespace": namespace,
        },
        dry_run=dry_run,
    )
    if dry_run:
        return
    scheduled_job = api.create_scheduled_job(
        image=image,
        command=command,
        schedule=schedule,
        suspend=suspend,
        concurrency=concurrency,
        env=env_map,
        secrets=secrets_map,
        labels=labels_map,
        volumes=volumes,
        flavor=flavor,
        timeout=timeout,
        expose=expose,
        resource_group_id=resource_group_id,
        namespace=namespace,
    )
    out.result("Scheduled Job created", id=scheduled_job.id, name=(scheduled_job.job_spec.labels or {}).get("name"))
    if not _has_explicit_name(name, label):
        auto_name = (scheduled_job.job_spec.labels or {}).get("name")
        out.hint(
            f"Scheduled Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs scheduled labels {scheduled_job.owner.name}/{scheduled_job.id} --name NAME` to rename it."
        )
    out.hint(f"Use `hf jobs scheduled inspect {scheduled_job.owner.name}/{scheduled_job.id}` to view its details.")


@scheduled_app.command("list | ls | ps", examples=["hf jobs scheduled ls"])
def scheduled_ps(
    all: Annotated[
        bool,
        Option(
            "-a",
            "--all",
            help="Show all scheduled Jobs (default hides suspended)",
        ),
    ] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    filter: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter",
            help="Filter output based on conditions provided (format: key=value)",
        ),
    ] = None,
) -> None:
    """List scheduled Jobs"""
    api = get_hf_api(token=token)
    scheduled_jobs = api.list_scheduled_jobs(namespace=namespace)
    filters: list[tuple[str, str, str]] = []
    for f in filter or []:
        if "=" in f:
            key, value = f.split("=", 1)
            # Negate predicate in case of key!=value
            if key.endswith("!"):
                op = "!="
                key = key[:-1]
            else:
                op = "="
            filters.append((key.lower(), op, value.lower()))
        else:
            out.warning(f"Ignoring invalid filter format '{f}'. Use key=value format.")

    # Filter scheduled jobs (operating on ScheduledJobInfo objects to preserve existing filter behavior)
    filtered_jobs = []
    for scheduled_job in scheduled_jobs:
        suspend = scheduled_job.suspend or False
        if not all and suspend:
            continue
        image_or_space = scheduled_job.job_spec.docker_image or "N/A"
        cmd = scheduled_job.job_spec.command or []
        command_str = " ".join(cmd) if cmd else "N/A"
        job_name = (scheduled_job.job_spec.labels or {}).get("name") or "N/A"
        props = {
            "id": scheduled_job.id,
            "name": job_name,
            "image": image_or_space,
            "suspend": str(suspend),
            "command": command_str,
        }
        if not _matches_filters(props, filters):
            continue
        filtered_jobs.append(scheduled_job)

    # Build display items. Augment with curated columns.
    items: list[dict[str, Any]] = []
    for sj in filtered_jobs:
        item = _dataclass_to_dict(sj)
        job_spec = item.get("job_spec") or {}
        status_dict = item.get("status") or {}
        last_job = status_dict.get("last_job")
        cmd = job_spec.get("command") or []
        item["name"] = (job_spec.get("labels") or {}).get("name") or "N/A"
        item["image/space"] = job_spec.get("docker_image") or "N/A"
        item["command"] = " ".join(cmd) if cmd else "N/A"
        item["last_run"] = last_job["at"][:19].replace("T", " ") if last_job and last_job.get("at") else "N/A"
        item["next_run"] = (
            status_dict["next_job_run_at"][:19].replace("T", " ") if status_dict.get("next_job_run_at") else "N/A"
        )
        item["suspend"] = item.get("suspend") or False
        items.append(item)

    out.table(
        items,
        headers=["id", "name", "schedule", "image/space", "command", "last_run", "next_run", "suspend"],
        id_key="id",
    )
    if not items and filters:
        filters_msg = ", ".join(f"{k}{o}{v}" for k, o, v in filters)
        out.text(f"No scheduled jobs matched filters: {filters_msg}")
    if items:
        first_item_id = items[0]["id"]
        out.hint(f"Use `hf jobs scheduled inspect {first_item_id}` to view details about a scheduled job.")
        out.hint(f"Use `hf jobs scheduled trigger {first_item_id}` to trigger a scheduled job immediately.")


@scheduled_app.command("inspect", examples=["hf jobs scheduled inspect <id>"])
def scheduled_inspect(
    scheduled_job_ids: Annotated[
        list[str],
        Argument(
            help="Scheduled Job IDs to inspect (or 'namespace/scheduled_job_id')",
        ),
    ],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Display detailed information on one or more scheduled Jobs"""
    parsed_ids = []
    for job_id in scheduled_job_ids:
        job_id, namespace = _parse_namespace_from_job_id(job_id, namespace)
        parsed_ids.append(job_id)
    scheduled_job_ids = parsed_ids
    api = get_hf_api(token=token)
    scheduled_jobs = [
        api.inspect_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
        for scheduled_job_id in scheduled_job_ids
    ]
    out.table([_surface_name(_dataclass_to_dict(sj), labels=sj.job_spec.labels) for sj in scheduled_jobs], id_key="id")


@scheduled_app.command("delete", examples=["hf jobs scheduled delete <id>"])
def scheduled_delete(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Delete a scheduled Job."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    api.delete_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job deleted", id=scheduled_job_id)


@scheduled_app.command("suspend", examples=["hf jobs scheduled suspend <id>"])
def scheduled_suspend(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Suspend (pause) a scheduled Job."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    api.suspend_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job suspended", id=scheduled_job_id)
    out.hint(f"Use `hf jobs scheduled resume {scheduled_job_id}` to resume it.")


@scheduled_app.command("resume", examples=["hf jobs scheduled resume <id>"])
def scheduled_resume(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Resume (unpause) a scheduled Job."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    api.resume_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job resumed", id=scheduled_job_id)


@scheduled_app.command("trigger", examples=["hf jobs scheduled trigger <id>"])
def scheduled_trigger(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Trigger a scheduled Job to run immediately (does not change the schedule)."""
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    job = api.trigger_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
    out.result("Scheduled Job triggered", id=job.id, url=job.url)
    out.hint(f"Use `hf jobs logs -f {job.owner.name}/{job.id}` to stream logs.")


@scheduled_app.command(
    "labels",
    examples=[
        "hf jobs scheduled labels <id> --name daily-script",
        "hf jobs scheduled labels <id> --label env=prod --label team=ml",
        "hf jobs scheduled labels <id> --clear",
    ],
)
def scheduled_labels(
    scheduled_job_id: ScheduledJobIdArg,
    name: NameOpt = None,
    label: LabelsOpt = None,
    clear: Annotated[bool, Option("--clear", help="Remove all labels from the scheduled job.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Update labels on a scheduled Job. Passing --label replaces all existing labels; passing --name alone keeps them."""
    if not label and name is None and not clear:
        raise CLIError(
            "Please set a name with --name or at least one label with --label. To remove all labels, pass --clear."
        )
    if (label or name is not None) and clear:
        raise CLIError(
            "Cannot set a name or labels and clear them at the same time. Please use --name/--label or --clear, not both."
        )
    scheduled_job_id, namespace = _parse_namespace_from_job_id(scheduled_job_id, namespace)
    api = get_hf_api(token=token)
    if name is not None and not label:
        # Naming a scheduled Job should not wipe its existing labels: fetch them and merge the name in.
        current_labels = (
            api.inspect_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace).job_spec.labels or {}
        )
        labels = {**current_labels, "name": name}
    else:
        labels = _parse_labels_map(label, name=name) or {}
    scheduled_job = api.update_scheduled_job_labels(
        scheduled_job_id=scheduled_job_id, labels=labels, namespace=namespace
    )
    out.result("Labels updated", id=scheduled_job.id, name=labels.get("name"))


scheduled_uv_app = typer_factory(help="Schedule UV scripts on HF infrastructure.")
scheduled_app.add_group(scheduled_uv_app, name="uv")


@scheduled_uv_app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        'hf jobs scheduled uv run "0 0 * * *" --name daily-script script.py',
        'hf jobs scheduled uv run "0 0 * * *" script.py --with pandas',
    ],
)
def scheduled_uv_run(
    schedule: ScheduleArg,
    script: ScriptArg,
    script_args: ScriptArgsArg = None,
    suspend: SuspendOpt = None,
    concurrency: ConcurrencyOpt = None,
    image: ImageOpt = None,
    flavor: FlavorOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    name: NameOpt = None,
    label: LabelsOpt = None,
    volume: JobVolumesOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    timeout: TimeoutOpt = None,
    dry_run: DryRunOpt = False,
    expose: ExposeOpt = None,
    resource_group_id: ResourceGroupIdOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    with_: WithOpt = None,
    python: PythonOpt = None,
) -> None:
    """Run a UV script (local file or URL) on HF infrastructure"""
    api = get_hf_api(token=token)
    config = _resolve_uv_job_config(
        api=api,
        script=script,
        script_args=script_args,
        dependencies=with_,
        image=image,
        flavor=flavor,
        python=python,
        env=env,
        env_file=env_file,
        secrets=secrets,
        secrets_file=secrets_file,
        timeout=timeout,
        name=name,
        label=label,
        volume=volume,
        namespace=namespace,
        dry_run=dry_run,
    )
    _print_job_summary(
        {
            "schedule": schedule,
            "script": script,
            "args": shlex.join(config.script_args),
            "with": " ".join(with_ or []),
            "image": config.image or DEFAULT_UV_IMAGE,
            "flavor": config.flavor or JobHardware.CPU_BASIC.value,
            "python": config.python,
            "timeout": config.timeout,
            "env": _format_env(config.env, from_script=config.from_script),
            "secrets": _format_secrets(config.secrets, from_script=config.from_script),
            "volumes": _format_volumes(config.volume_specs, from_script=config.from_script),
            "labels": _format_labels(config.labels, from_script=config.from_script),
            "expose": " ".join(str(port) for port in expose or []),
            "namespace": config.namespace,
        },
        from_script=config.from_script,
        dry_run=dry_run,
    )
    if dry_run:
        return
    job = api.create_scheduled_uv_job(
        script=config.script,
        script_args=config.script_args,
        schedule=schedule,
        suspend=suspend,
        concurrency=concurrency,
        dependencies=with_,
        python=config.python,
        image=config.image,
        env=config.env,
        secrets=config.secrets,
        labels=config.labels,
        volumes=config.volumes,
        flavor=config.flavor,
        timeout=config.timeout,
        expose=expose,
        resource_group_id=resource_group_id,
        namespace=config.namespace,
    )
    out.result("Scheduled Job created", id=job.id, name=(job.job_spec.labels or {}).get("name"))
    if not _has_explicit_name(name, label, config.from_script):
        auto_name = (job.job_spec.labels or {}).get("name")
        out.hint(
            f"Scheduled Job auto-named '{auto_name}'. Pass `--name` or run "
            f"`hf jobs scheduled labels {job.owner.name}/{job.id} --name NAME` to rename it."
        )
    out.hint(f"Use `hf jobs scheduled inspect {job.owner.name}/{job.id}` to view its details.")


### UTILS


def _surface_name(item: dict[str, Any], *, labels: dict[str, str] | None) -> dict[str, Any]:
    """Promote the `name` label to a top-level `name` field for display.

    The `name` is kept inside `labels` too (the dataclasses and API payloads are unchanged); this
    only makes it a first-class column/field in command outputs. No-op when there is no name.
    """
    name = (labels or {}).get("name")
    if name is None:
        return item
    return {"name": name, **item}


def _has_explicit_name(name: str | None, label: list[str] | None, from_script: Collection[str] = ()) -> bool:
    """Whether the Job was named on purpose: via `--name`, a `name=` label, or the script's `[tool.hf-jobs]` table."""
    return (
        name is not None
        or "labels.name" in from_script
        or any(item.split("=", 1)[0] == "name" for item in label or [])
    )


def _parse_labels_map(labels: list[str] | None, *, name: str | None = None) -> dict[str, str] | None:
    """Parse label key-value pairs from CLI arguments.

    Args:
        labels: List of label strings in KEY=VALUE format. If KEY only, then VALUE is set to empty string.

    Returns:
        Dictionary mapping label keys to values, or None if no labels or name provided.
    """
    if not labels and name is None:
        return None
    labels_map: dict[str, str] = {}
    for label_var in labels or []:
        key, value = label_var.split("=", 1) if "=" in label_var else (label_var, "")
        labels_map[key] = value
    if name is not None:
        if "name" in labels_map:
            raise CLIError("--name and --label name=... cannot both be provided.")
        labels_map["name"] = name
    return labels_map


def _tabulate(rows: list[list[str | int]], headers: list[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    terminal_width = max(shutil.get_terminal_size().columns, len(headers) * 12)
    while len(headers) + sum(col_widths) > terminal_width:
        col_to_minimize = col_widths.index(max(col_widths))
        col_widths[col_to_minimize] //= 2
        if len(headers) + sum(col_widths) <= terminal_width:
            col_widths[col_to_minimize] = terminal_width - sum(col_widths) - len(headers) + col_widths[col_to_minimize]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        row_format_args = [
            str(x)[: col_width - 3] + "..." if len(str(x)) > col_width else str(x)
            for x, col_width in zip(row, col_widths)
        ]
        lines.append(row_format.format(*row_format_args))
    return "\n".join(lines)


T = TypeVar("T")


def _write_generator_to_queue(queue: Queue[T], func: Callable[..., Iterable[T]], kwargs: dict) -> None:
    for result in func(**kwargs):
        queue.put(result)


def iflatmap_unordered(
    pool: multiprocessing.pool.ThreadPool,
    func: Callable[..., Iterable[T]],
    *,
    kwargs_list: list[dict],
) -> Iterable[T]:
    """
    Takes a function that returns an iterable of items, and run it in parallel using threads to return the flattened iterable of items as they arrive.

    This is inspired by those three `map()` variants, and is the mix of all three:

    * `imap()`: like `map()` but returns an iterable instead of a list of results
    * `imap_unordered()`: like `imap()` but the output is sorted by time of arrival
    * `flatmap()`: like `map()` but given a function which returns a list, `flatmap()` returns the flattened list that is the concatenation of all the output lists
    """
    queue: Queue[T] = Queue()
    async_results = [pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_list]
    try:
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready() for async_result in async_results) and queue.empty():
                    break
    except KeyboardInterrupt:
        pass
    finally:
        # we get the result in case there's an error to raise
        try:
            [async_result.get(timeout=0.05) for async_result in async_results]
        except multiprocessing.TimeoutError:
            pass
