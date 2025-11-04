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
"""Contains commands to interact with jobs on the Hugging Face Hub.

Usage:
    # run a job
    hf jobs run <image> <command>

    # List running or completed jobs
    hf jobs ps [-a] [-f key=value] [--format TEMPLATE]

    # Stream logs from a job
    hf jobs logs <job-id>

    # Inspect detailed information about a job
    hf jobs inspect <job-id>

    # Cancel a running job
    hf jobs cancel <job-id>

    # Run a UV script
    hf jobs uv run <script>

    # Schedule a job
    hf jobs scheduled run <schedule> <image> <command>

    # List scheduled jobs
    hf jobs scheduled ps [-a] [-f key=value] [--format TEMPLATE]

    # Inspect a scheduled job
    hf jobs scheduled inspect <scheduled_job_id>

    # Suspend a scheduled job
    hf jobs scheduled suspend <scheduled_job_id>

    # Resume a scheduled job
    hf jobs scheduled resume <scheduled_job_id>

    # Delete a scheduled job
    hf jobs scheduled delete <scheduled_job_id>

"""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Dict, Optional, Union

import typer

from huggingface_hub import SpaceHardware, get_token
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import logging
from huggingface_hub.utils._dotenv import load_dotenv

from ._cli_utils import TokenOpt, get_hf_api, typer_factory


logger = logging.get_logger(__name__)

SUGGESTED_FLAVORS = [item.value for item in SpaceHardware if item.value != "zero-a10g"]

# Common job-related options
ImageArg = Annotated[
    str,
    typer.Argument(
        help="The Docker image to use.",
    ),
]

ImageOpt = Annotated[
    Optional[str],
    typer.Option(
        help="Use a custom Docker image with `uv` installed.",
    ),
]

FlavorOpt = Annotated[
    Optional[SpaceHardware],
    typer.Option(
        help=f"Flavor for the hardware, as in HF Spaces. Defaults to `cpu-basic`. Possible values: {', '.join(SUGGESTED_FLAVORS)}.",
    ),
]

EnvOpt = Annotated[
    Optional[list[str]],
    typer.Option(
        "-e",
        "--env",
        help="Set environment variables. E.g. --env ENV=value",
    ),
]

SecretsOpt = Annotated[
    Optional[list[str]],
    typer.Option(
        "-s",
        "--secrets",
        help="Set secret environment variables. E.g. --secrets SECRET=value or `--secrets HF_TOKEN` to pass your Hugging Face token.",
    ),
]

EnvFileOpt = Annotated[
    Optional[str],
    typer.Option(
        "--env-file",
        help="Read in a file of environment variables.",
    ),
]

SecretsFileOpt = Annotated[
    Optional[str],
    typer.Option(
        help="Read in a file of secret environment variables.",
    ),
]

TimeoutOpt = Annotated[
    Optional[str],
    typer.Option(
        help="Max duration: int/float with s (seconds, default), m (minutes), h (hours) or d (days).",
    ),
]

DetachOpt = Annotated[
    bool,
    typer.Option(
        "-d",
        "--detach",
        help="Run the Job in the background and print the Job ID.",
    ),
]

NamespaceOpt = Annotated[
    Optional[str],
    typer.Option(
        help="The namespace where the job will be running. Defaults to the current user's namespace.",
    ),
]

WithOpt = Annotated[
    Optional[list[str]],
    typer.Option(
        "--with",
        help="Run with the given packages installed",
    ),
]

PythonOpt = Annotated[
    Optional[str],
    typer.Option(
        "-p",
        "--python",
        help="The Python interpreter to use for the run environment",
    ),
]

SuspendOpt = Annotated[
    Optional[bool],
    typer.Option(
        help="Suspend (pause) the scheduled Job",
    ),
]

ConcurrencyOpt = Annotated[
    Optional[bool],
    typer.Option(
        help="Allow multiple instances of this Job to run concurrently",
    ),
]

ScheduleArg = Annotated[
    str,
    typer.Argument(
        help="One of annually, yearly, monthly, weekly, daily, hourly, or a CRON schedule expression.",
    ),
]

ScriptArg = Annotated[
    str,
    typer.Argument(
        help="UV script to run (local file or URL)",
    ),
]

ScriptArgsArg = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="Arguments for the script",
    ),
]

CommandArg = Annotated[
    list[str],
    typer.Argument(
        help="The command to run.",
    ),
]

JobIdArg = Annotated[
    str,
    typer.Argument(
        help="Job ID",
    ),
]

ScheduledJobIdArg = Annotated[
    str,
    typer.Argument(
        help="Scheduled Job ID",
    ),
]

RepoOpt = Annotated[
    Optional[str],
    typer.Option(
        help="Repository name for the script (creates ephemeral if not specified)",
    ),
]


jobs_cli = typer_factory(help="Run and manage Jobs on the Hub.")


@jobs_cli.command("run", help="Run a Job")
def jobs_run(
    image: ImageArg,
    command: CommandArg,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    detach: DetachOpt = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    env_map: dict[str, Optional[str]] = {}
    if env_file:
        env_map.update(load_dotenv(Path(env_file).read_text(), environ=os.environ.copy()))
    for env_value in env or []:
        env_map.update(load_dotenv(env_value, environ=os.environ.copy()))

    secrets_map: dict[str, Optional[str]] = {}
    extended_environ = _get_extended_environ()
    if secrets_file:
        secrets_map.update(load_dotenv(Path(secrets_file).read_text(), environ=extended_environ))
    for secret in secrets or []:
        secrets_map.update(load_dotenv(secret, environ=extended_environ))

    api = get_hf_api(token=token)
    job = api.run_job(
        image=image,
        command=command,
        env=env_map,
        secrets=secrets_map,
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
    )
    # Always print the job ID to the user
    print(f"Job started with ID: {job.id}")
    print(f"View at: {job.url}")

    if detach:
        return
    # Now let's stream the logs
    for log in api.fetch_job_logs(job_id=job.id):
        print(log)


@jobs_cli.command("logs", help="Fetch the logs of a Job")
def jobs_logs(
    job_id: JobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    for log in api.fetch_job_logs(job_id=job_id, namespace=namespace):
        print(log)


def _matches_filters(job_properties: dict[str, str], filters: dict[str, str]) -> bool:
    """Check if scheduled job matches all specified filters."""
    for key, pattern in filters.items():
        # Check if property exists
        if key not in job_properties:
            return False
        # Support pattern matching with wildcards
        if "*" in pattern or "?" in pattern:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            if not re.search(f"^{regex_pattern}$", job_properties[key], re.IGNORECASE):
                return False
        # Simple substring matching
        elif pattern.lower() not in job_properties[key].lower():
            return False
    return True


def _print_output(rows: list[list[Union[str, int]]], headers: list[str], fmt: Optional[str]) -> None:
    """Print output according to the chosen format."""
    if fmt:
        # Use custom template if provided
        template = fmt
        for row in rows:
            line = template
            for i, field in enumerate(["id", "image", "command", "created", "status"]):
                placeholder = f"{{{{.{field}}}}}"
                if placeholder in line:
                    line = line.replace(placeholder, str(row[i]))
            print(line)
    else:
        # Default tabular format
        print(_tabulate(rows, headers=headers))


@jobs_cli.command("ps", help="List Jobs")
def jobs_ps(
    all: Annotated[
        bool,
        typer.Option(
            "-a",
            "--all",
            help="Show all Jobs (default shows just running)",
        ),
    ] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    filter: Annotated[
        Optional[list[str]],
        typer.Option(
            "-f",
            "--filter",
            help="Filter output based on conditions provided (format: key=value)",
        ),
    ] = None,
    format: Annotated[
        Optional[str],
        typer.Option(
            help="Format output using a custom template",
        ),
    ] = None,
) -> None:
    try:
        api = get_hf_api(token=token)
        # Fetch jobs data
        jobs = api.list_jobs(namespace=namespace)
        # Define table headers
        table_headers = ["JOB ID", "IMAGE/SPACE", "COMMAND", "CREATED", "STATUS"]
        rows: list[list[Union[str, int]]] = []

        filters: dict[str, str] = {}
        for f in filter or []:
            if "=" in f:
                key, value = f.split("=", 1)
                filters[key.lower()] = value
            else:
                print(f"Warning: Ignoring invalid filter format '{f}'. Use key=value format.")
        # Process jobs data
        for job in jobs:
            # Extract job data for filtering
            status = job.status.stage if job.status else "UNKNOWN"
            if not all and status not in ("RUNNING", "UPDATING"):
                # Skip job if not all jobs should be shown and status doesn't match criteria
                continue
            # Extract job data for output
            job_id = job.id

            # Extract image or space information
            image_or_space = job.docker_image or "N/A"

            # Extract and format command
            cmd = job.command or []
            command_str = " ".join(cmd) if cmd else "N/A"

            # Extract creation time
            created_at = job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A"

            # Create a dict with all job properties for filtering
            props = {"id": job_id, "image": image_or_space, "status": status.lower(), "command": command_str}
            if not _matches_filters(props, filters):
                continue

            # Create row
            rows.append([job_id, image_or_space, command_str, created_at, status])

        # Handle empty results
        if not rows:
            filters_msg = (
                f" matching filters: {', '.join([f'{k}={v}' for k, v in filters.items()])}" if filters else ""
            )
            print(f"No jobs found{filters_msg}")
            return
        # Apply custom format if provided or use default tabular format
        _print_output(rows, table_headers, format)

    except HfHubHTTPError as e:
        print(f"Error fetching jobs data: {e}")
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing jobs data: {e}")
    except Exception as e:
        print(f"Unexpected error - {type(e).__name__}: {e}")


@jobs_cli.command("inspect", help="Display detailed information on one or more Jobs")
def jobs_inspect(
    job_ids: Annotated[
        list[str],
        typer.Argument(
            help="The jobs to inspect",
        ),
    ],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    jobs = [api.inspect_job(job_id=job_id, namespace=namespace) for job_id in job_ids]
    print(json.dumps([asdict(job) for job in jobs], indent=4, default=str))


@jobs_cli.command("cancel", help="Cancel a Job")
def jobs_cancel(
    job_id: JobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    api.cancel_job(job_id=job_id, namespace=namespace)


uv_app = typer_factory(help="Run UV scripts (Python with inline dependencies) on HF infrastructure")
jobs_cli.add_typer(uv_app, name="uv")


@uv_app.command("run", help="Run a UV script (local file or URL) on HF infrastructure")
def jobs_uv_run(
    script: ScriptArg,
    script_args: ScriptArgsArg = None,
    image: ImageOpt = None,
    repo: RepoOpt = None,
    flavor: FlavorOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    timeout: TimeoutOpt = None,
    detach: DetachOpt = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    with_: WithOpt = None,
    python: PythonOpt = None,
) -> None:
    env_map: dict[str, Optional[str]] = {}
    if env_file:
        env_map.update(load_dotenv(Path(env_file).read_text(), environ=os.environ.copy()))
    for env_value in env or []:
        env_map.update(load_dotenv(env_value, environ=os.environ.copy()))
    secrets_map: dict[str, Optional[str]] = {}
    extended_environ = _get_extended_environ()
    if secrets_file:
        secrets_map.update(load_dotenv(Path(secrets_file).read_text(), environ=extended_environ))
    for secret in secrets or []:
        secrets_map.update(load_dotenv(secret, environ=extended_environ))

    api = get_hf_api(token=token)
    job = api.run_uv_job(
        script=script,
        script_args=script_args or [],
        dependencies=with_,
        python=python,
        image=image,
        env=env_map,
        secrets=secrets_map,
        flavor=flavor,  # type: ignore[arg-type]
        timeout=timeout,
        namespace=namespace,
        _repo=repo,
    )
    # Always print the job ID to the user
    print(f"Job started with ID: {job.id}")
    print(f"View at: {job.url}")
    if detach:
        return
    # Now let's stream the logs
    for log in api.fetch_job_logs(job_id=job.id):
        print(log)


scheduled_app = typer_factory(help="Create and manage scheduled Jobs on the Hub.")
jobs_cli.add_typer(scheduled_app, name="scheduled")


@scheduled_app.command("run", help="Schedule a Job")
def scheduled_run(
    schedule: ScheduleArg,
    image: ImageArg,
    command: CommandArg,
    suspend: SuspendOpt = None,
    concurrency: ConcurrencyOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    env_map: dict[str, Optional[str]] = {}
    if env_file:
        env_map.update(load_dotenv(Path(env_file).read_text(), environ=os.environ.copy()))
    for env_value in env or []:
        env_map.update(load_dotenv(env_value, environ=os.environ.copy()))
    secrets_map: dict[str, Optional[str]] = {}
    extended_environ = _get_extended_environ()
    if secrets_file:
        secrets_map.update(load_dotenv(Path(secrets_file).read_text(), environ=extended_environ))
    for secret in secrets or []:
        secrets_map.update(load_dotenv(secret, environ=extended_environ))

    api = get_hf_api(token=token)
    scheduled_job = api.create_scheduled_job(
        image=image,
        command=command,
        schedule=schedule,
        suspend=suspend,
        concurrency=concurrency,
        env=env_map,
        secrets=secrets_map,
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
    )
    print(f"Scheduled Job created with ID: {scheduled_job.id}")


@scheduled_app.command("ps", help="List scheduled Jobs")
def scheduled_ps(
    all: Annotated[
        bool,
        typer.Option(
            "-a",
            "--all",
            help="Show all scheduled Jobs (default hides suspended)",
        ),
    ] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    filter: Annotated[
        Optional[list[str]],
        typer.Option(
            "-f",
            "--filter",
            help="Filter output based on conditions provided (format: key=value)",
        ),
    ] = None,
    format: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            help="Format output using a custom template",
        ),
    ] = None,
) -> None:
    try:
        api = get_hf_api(token=token)
        scheduled_jobs = api.list_scheduled_jobs(namespace=namespace)
        table_headers = ["ID", "SCHEDULE", "IMAGE/SPACE", "COMMAND", "LAST RUN", "NEXT RUN", "SUSPEND"]
        rows: list[list[Union[str, int]]] = []
        filters: dict[str, str] = {}
        for f in filter or []:
            if "=" in f:
                key, value = f.split("=", 1)
                filters[key.lower()] = value
            else:
                print(f"Warning: Ignoring invalid filter format '{f}'. Use key=value format.")

        for scheduled_job in scheduled_jobs:
            suspend = scheduled_job.suspend or False
            if not all and suspend:
                continue
            sj_id = scheduled_job.id
            schedule = scheduled_job.schedule or "N/A"
            image_or_space = scheduled_job.job_spec.docker_image or "N/A"
            cmd = scheduled_job.job_spec.command or []
            command_str = " ".join(cmd) if cmd else "N/A"
            last_job_at = (
                scheduled_job.status.last_job.at.strftime("%Y-%m-%d %H:%M:%S")
                if scheduled_job.status.last_job
                else "N/A"
            )
            next_job_run_at = (
                scheduled_job.status.next_job_run_at.strftime("%Y-%m-%d %H:%M:%S")
                if scheduled_job.status.next_job_run_at
                else "N/A"
            )
            props = {"id": sj_id, "image": image_or_space, "suspend": str(suspend), "command": command_str}
            if not _matches_filters(props, filters):
                continue
            rows.append([sj_id, schedule, image_or_space, command_str, last_job_at, next_job_run_at, suspend])

        if not rows:
            filters_msg = (
                f" matching filters: {', '.join([f'{k}={v}' for k, v in filters.items()])}" if filters else ""
            )
            print(f"No scheduled jobs found{filters_msg}")
            return
        _print_output(rows, table_headers, format)

    except HfHubHTTPError as e:
        print(f"Error fetching scheduled jobs data: {e}")
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing scheduled jobs data: {e}")
    except Exception as e:
        print(f"Unexpected error - {type(e).__name__}: {e}")


@scheduled_app.command("inspect", help="Display detailed information on one or more scheduled Jobs")
def scheduled_inspect(
    scheduled_job_ids: Annotated[
        list[str],
        typer.Argument(
            help="The scheduled jobs to inspect",
        ),
    ],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    scheduled_jobs = [
        api.inspect_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)
        for scheduled_job_id in scheduled_job_ids
    ]
    print(json.dumps([asdict(scheduled_job) for scheduled_job in scheduled_jobs], indent=4, default=str))


@scheduled_app.command("delete", help="Delete a scheduled Job")
def scheduled_delete(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    api.delete_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)


@scheduled_app.command("suspend", help="Suspend (pause) a scheduled Job")
def scheduled_suspend(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    api.suspend_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)


@scheduled_app.command("resume", help="Resume (unpause) a scheduled Job")
def scheduled_resume(
    scheduled_job_id: ScheduledJobIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    api = get_hf_api(token=token)
    api.resume_scheduled_job(scheduled_job_id=scheduled_job_id, namespace=namespace)


scheduled_uv_app = typer_factory(help="Schedule UV scripts on HF infrastructure")
scheduled_app.add_typer(scheduled_uv_app, name="uv")


@scheduled_uv_app.command("run", help="Run a UV script (local file or URL) on HF infrastructure")
def scheduled_uv_run(
    schedule: ScheduleArg,
    script: ScriptArg,
    script_args: ScriptArgsArg = None,
    suspend: SuspendOpt = None,
    concurrency: ConcurrencyOpt = None,
    image: ImageOpt = None,
    repo: RepoOpt = None,
    flavor: FlavorOpt = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    timeout: TimeoutOpt = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
    with_: WithOpt = None,
    python: PythonOpt = None,
) -> None:
    env_map: dict[str, Optional[str]] = {}
    if env_file:
        env_map.update(load_dotenv(Path(env_file).read_text(), environ=os.environ.copy()))
    for env_value in env or []:
        env_map.update(load_dotenv(env_value, environ=os.environ.copy()))
    secrets_map: dict[str, Optional[str]] = {}
    extended_environ = _get_extended_environ()
    if secrets_file:
        secrets_map.update(load_dotenv(Path(secrets_file).read_text(), environ=extended_environ))
    for secret in secrets or []:
        secrets_map.update(load_dotenv(secret, environ=extended_environ))

    api = get_hf_api(token=token)
    job = api.create_scheduled_uv_job(
        script=script,
        script_args=script_args or [],
        schedule=schedule,
        suspend=suspend,
        concurrency=concurrency,
        dependencies=with_,
        python=python,
        image=image,
        env=env_map,
        secrets=secrets_map,
        flavor=flavor,  # type: ignore[arg-type]
        timeout=timeout,
        namespace=namespace,
        _repo=repo,
    )
    print(f"Scheduled Job created with ID: {job.id}")


### UTILS


def _tabulate(rows: list[list[Union[str, int]]], headers: list[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    terminal_width = max(os.get_terminal_size().columns, len(headers) * 12)
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


def _get_extended_environ() -> Dict[str, str]:
    extended_environ = os.environ.copy()
    if (token := get_token()) is not None:
        extended_environ["HF_TOKEN"] = token
    return extended_environ
