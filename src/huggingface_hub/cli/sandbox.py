# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Contains commands to run and manage sandboxes on Hugging Face Jobs.

Two ways to get a sandbox:

    # One dedicated sandbox (a full VM — GPU, untrusted code, exposed ports)
    hf sandbox create [IMAGE]

    # Many cheap shared sandboxes packed into host VMs (CPU fan-out, RL rollouts).
    # Define a pool once, then spawn sandboxes from it on demand:
    hf sandbox pool create --image python:3.12 --flavor cpu-basic   # -> pool id
    hf sandbox pool spawn <pool_id>                                 # reuse/boot a host

Both kinds share the same commands:

    hf sandbox exec <sandbox_id> -- python -c "print('hi')"
    hf sandbox cp local.txt <sandbox_id>:/tmp/remote.txt
    hf sandbox ls
    hf sandbox ps <sandbox_id>
    hf sandbox url <sandbox_id> <port>
    hf sandbox kill <sandbox_id>
"""

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Iterator

import typer

from huggingface_hub import constants
from huggingface_hub._sandbox import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_IMAGE,
    DEFAULT_SANDBOXES_PER_HOST,
    HOST_LABEL,
    POOL_LABEL,
    SANDBOX_LABEL,
    SHARED_ID_SEP,
    Sandbox,
    SandboxPool,
    _connect_host,
    _split_sandbox_id,
)
from huggingface_hub.errors import CLIError, SandboxError

from ._cli_utils import (
    EnvFileOpt,
    EnvOpt,
    SecretsFileOpt,
    SecretsOpt,
    TokenOpt,
    VolumesOpt,
    get_hf_api,
    parse_env_map,
    parse_volumes,
    typer_factory,
)
from ._output import out
from .jobs import ExposeOpt, FlavorOpt, NamespaceOpt, TimeoutOpt


sandbox_cli = typer_factory(help="Run and manage sandboxes on Hugging Face Jobs.")

SandboxIdArg = Annotated[str, typer.Argument(help="The sandbox id (as printed by `hf sandbox create`).")]


@contextmanager
def _connect(sandbox_id: str, *, namespace: str | None, token: str | None) -> Iterator[Sandbox]:
    """Reattach to a sandbox and close the HTTP client when the command is done.

    Closing matters for one-shot CLI commands: without it the httpx client (and its
    connection pool) would leak until interpreter shutdown.
    """
    sandbox = Sandbox.connect(sandbox_id, namespace=namespace, token=token)
    try:
        yield sandbox
    finally:
        sandbox.close()


@sandbox_cli.command(
    "create",
    examples=[
        "hf sandbox create",
        "hf sandbox create ubuntu:24.04",
        "hf sandbox create --flavor a10g-small --timeout 1h",
        "hf sandbox create --expose 8080",
    ],
)
def sandbox_create(
    image: Annotated[str, typer.Argument(help="Docker image (needs /bin/sh).")] = DEFAULT_IMAGE,
    flavor: FlavorOpt = None,
    timeout: TimeoutOpt = None,
    idle_timeout: Annotated[
        str | None,
        typer.Option(help="Auto-terminate after this much inactivity (e.g. '10m'). Defaults to 10m."),
    ] = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    volume: VolumesOpt = None,
    expose: ExposeOpt = None,
    namespace: NamespaceOpt = None,
    forward_hf_token: Annotated[
        bool, typer.Option("--forward-hf-token", help="Inject your HF token as HF_TOKEN in the sandbox.")
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Create one dedicated sandbox (a full VM).

    For many cheap CPU sandboxes packed into shared host VMs, define a pool with
    `hf sandbox pool create` and spawn from it with `hf sandbox pool spawn`.
    """
    start = time.time()
    idle = idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT
    sandbox = Sandbox.create(
        image=image,
        flavor=flavor or "cpu-basic",
        timeout=timeout,
        idle_timeout=idle,
        env=parse_env_map(env, env_file),
        secrets=parse_env_map(secrets, secrets_file),
        volumes=parse_volumes(volume),
        expose=expose,
        namespace=namespace,
        forward_hf_token=forward_hf_token,
        token=token,
    )
    # Release the HTTP client (the sandbox keeps running); this one-shot command reports the id
    # and exits, later commands reattach with their own connection.
    sandbox.close()
    out.result("Sandbox ready", id=sandbox.id, image=image, elapsed=f"{time.time() - start:.1f}s")
    out.hint(f"Run a command with `hf sandbox exec {sandbox.id} -- echo hello`.")
    out.hint(f"Terminate it with `hf sandbox kill {sandbox.id}`.")


@sandbox_cli.command("ls | list", examples=["hf sandbox ls"])
def sandbox_ls(
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List your running sandboxes (dedicated and shared)."""
    api = get_hf_api(token=token)
    rows = []
    hosts = []
    for job in api.list_jobs(namespace=namespace):
        labels = job.labels or {}
        if not labels.get(SANDBOX_LABEL) or job.status.stage != "RUNNING":
            continue
        created = job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A"
        image = job.docker_image or job.space_id
        if labels.get(HOST_LABEL):
            hosts.append((job, image, job.flavor, created))
        else:
            rows.append({"id": job.id, "kind": "dedicated", "image": image, "flavor": job.flavor, "created": created})

    # For each host job, enumerate its packed sandboxes via the server.
    for job, image, flavor, created in hosts:
        try:
            server = _connect_host(api, job.id, namespace=namespace)
        except SandboxError:
            continue  # host still starting up or unreachable; skip its sandboxes
        try:
            for item in server.request("GET", "/v1/sandboxes").json():
                rows.append(
                    {
                        "id": f"{job.id}{SHARED_ID_SEP}{item['id']}",
                        "kind": "shared",
                        "image": image,
                        "flavor": flavor,
                        "created": created,
                    }
                )
        finally:
            server.close()

    out.table(rows, id_key="id")
    if not rows:
        out.hint("Create one with `hf sandbox create` (or define a pool with `hf sandbox pool create`).")


@sandbox_cli.command(
    "exec",
    context_settings={"ignore_unknown_options": True},
    examples=[
        'hf sandbox exec <sandbox_id> -- python -c "print(42)"',
        "hf sandbox exec -w /app <sandbox_id> -- pytest -x",
    ],
)
def sandbox_exec(
    sandbox_id: SandboxIdArg,
    command: Annotated[list[str], typer.Argument(help="The command to run.")],
    workdir: Annotated[str | None, typer.Option("-w", "--workdir", help="Working directory.")] = None,
    env: EnvOpt = None,
    env_file: EnvFileOpt = None,
    exec_timeout: Annotated[
        float | None, typer.Option("--timeout", help="Kill the command after this many seconds.")
    ] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Run a command in a sandbox, streaming output. Exits with the command's exit code."""

    def write_stdout(data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()

    def write_stderr(data: str) -> None:
        sys.stderr.write(data)
        sys.stderr.flush()

    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        result = sandbox.run(
            list(command),
            env=parse_env_map(env, env_file),
            cwd=workdir,
            timeout=exec_timeout,
            on_stdout=write_stdout,
            on_stderr=write_stderr,
            check=False,
        )
    if result.timed_out:
        out.error(f"Command timed out after {exec_timeout}s.")
        raise typer.Exit(code=result.exit_code or 124)  # 124: conventional timeout exit code
    if result.exit_code != 0:
        raise typer.Exit(code=result.exit_code if result.exit_code is not None else 1)


@sandbox_cli.command("ps", examples=["hf sandbox ps <sandbox_id>"])
def sandbox_ps(
    sandbox_id: SandboxIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List background processes running in a sandbox."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        rows = [
            {
                "pid": proc.pid,
                "tag": proc.tag or "",
                "status": "running" if proc.running else f"exited ({proc.exit_code})",
                "command": proc.cmd if len(proc.cmd) <= 60 else proc.cmd[:57] + "...",
            }
            for proc in sandbox.processes()
        ]
    out.table(rows, id_key="pid")


@sandbox_cli.command(
    "cp",
    examples=[
        "hf sandbox cp data.csv <sandbox_id>:/data/data.csv",
        "hf sandbox cp <sandbox_id>:/app/result.json result.json",
    ],
)
def sandbox_cp(
    src: Annotated[str, typer.Argument(help="Source: a local path or <sandbox_id>:<path>.")],
    dst: Annotated[str, typer.Argument(help="Destination: a local path or <sandbox_id>:<path>.")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Copy a file between the local machine and a sandbox (docker-style)."""

    def parse(ref: str) -> tuple[str | None, str]:
        # Only treat as a sandbox ref when the part before ':' looks like a sandbox id
        # (more than one char): this leaves local paths and Windows drive letters like
        # 'C:\data\file.csv' or 'C:/data/file.csv' (single-letter prefix) untouched.
        if ":" in ref and not ref.startswith((".", "/", "~")):
            sandbox_id, path = ref.split(":", 1)
            if len(sandbox_id) > 1:
                return sandbox_id, path
        return None, ref

    src_sandbox, src_path = parse(src)
    dst_sandbox, dst_path = parse(dst)
    if (src_sandbox is None) == (dst_sandbox is None):
        raise CLIError("Exactly one of SRC and DST must be a sandbox path (<sandbox_id>:<path>).")
    if src_sandbox is not None:
        with _connect(src_sandbox, namespace=namespace, token=token) as sandbox:
            sandbox.files.download(src_path, dst_path)
    else:
        assert dst_sandbox is not None
        with _connect(dst_sandbox, namespace=namespace, token=token) as sandbox:
            sandbox.files.upload(src_path, dst_path)
    out.result("Copied", src=src, dst=dst)


@sandbox_cli.command("url", examples=["hf sandbox url <sandbox_id> 8080"])
def sandbox_url(
    sandbox_id: SandboxIdArg,
    port: Annotated[int, typer.Argument(help="Container port (must have been exposed at creation).")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Print the public URL of an exposed sandbox port (dedicated sandboxes only)."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        url = sandbox.url(port)
    out.text(url)
    out.hint("Requests must include an HF token: `curl -H 'Authorization: Bearer ...' <url>`.")


@sandbox_cli.command(
    "kill",
    examples=[
        "hf sandbox kill <sandbox_id>",
        "hf sandbox kill <host_id>   # kills a whole shared host (all its sandboxes)",
        "hf sandbox kill --all",
    ],
)
def sandbox_kill(
    sandbox_id: Annotated[str | None, typer.Argument(help="The sandbox or host id to terminate.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Terminate every sandbox and host in the namespace.")] = False,
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Answer Yes to prompts automatically.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Terminate a sandbox, a whole shared host, or everything (--all)."""
    api = get_hf_api(token=token)

    if all_:
        jobs = [
            job
            for job in api.list_jobs(namespace=namespace)
            if (job.labels or {}).get(SANDBOX_LABEL) and job.status.stage == "RUNNING"
        ]
        if not jobs:
            out.text("No running sandboxes.")
            return
        out.confirm(f"Terminate {len(jobs)} sandbox job(s) (including shared hosts and all their sandboxes)?", yes=yes)
        for job in jobs:
            api.cancel_job(job_id=job.id, namespace=job.owner.name)
        out.result("Terminated", jobs=len(jobs))
        return

    if sandbox_id is None:
        raise CLIError("Provide a sandbox id, a host id, or --all.")

    sid, ns = _split_sandbox_id(sandbox_id, namespace)
    if SHARED_ID_SEP in sid:
        # One shared sandbox: remove it from its host (frees a slot; host keeps running).
        try:
            with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
                sandbox.kill()
        except SandboxError as e:
            raise CLIError(str(e)) from e
        out.result("Sandbox terminated", id=sandbox_id)
        return

    # A bare id: either a dedicated sandbox job or a shared host job.
    job = api.inspect_job(job_id=sid, namespace=ns)
    if (job.labels or {}).get(HOST_LABEL):
        out.confirm(f"Terminate shared host {sid} and all of its sandboxes?", yes=yes)
        api.cancel_job(job_id=job.id, namespace=job.owner.name)
        out.result("Host terminated", id=sid)
        return
    try:
        with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
            sandbox.kill()
    except SandboxError as e:
        raise CLIError(str(e)) from e
    out.result("Sandbox terminated", id=sandbox_id)


# --------------------------------------------------------------------- pool subgroup
#
# A "pool" is a saved set of sandbox options (image, flavor, env, ...). It is purely a
# local config — `pool create` starts no job and bills nothing. Spawning from it
# (`pool spawn <id>`) packs landlock sandboxes onto shared host VMs, reusing a warm host
# (discovered via the `hf-sandbox-pool=<id>` job label, so reuse works across processes
# and machines) before booting a new one.

pool_cli = typer_factory(help="Define sandbox pools and spawn cheap shared sandboxes from them.")
sandbox_cli.add_typer(pool_cli, name="pool")

_POOLS_DIR = Path(constants.HF_HOME) / "sandbox" / "pools"


def _pool_path(pool_id: str) -> Path:
    return _POOLS_DIR / f"{pool_id}.json"


def _save_pool(config: dict[str, Any]) -> None:
    _POOLS_DIR.mkdir(parents=True, exist_ok=True)
    # The config may hold secrets, so keep both the directory and the file private (0700/0600).
    os.chmod(_POOLS_DIR, 0o700)
    path = _pool_path(config["id"])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=2)


def _load_pool(pool_id: str) -> dict[str, Any]:
    path = _pool_path(pool_id)
    if not path.is_file():
        raise CLIError(f"Unknown pool '{pool_id}'. List pools with `hf sandbox pool ls`.")
    return json.loads(path.read_text())


@pool_cli.command(
    "create",
    examples=[
        "hf sandbox pool create",
        "hf sandbox pool create --image python:3.12 --flavor cpu-basic",
        "hf sandbox pool create --per-host 50 --secret OPENAI_API_KEY=sk-...",
    ],
)
def pool_create(
    image: Annotated[str, typer.Option("--image", help="Docker image for the hosts (needs /bin/sh).")] = DEFAULT_IMAGE,
    flavor: FlavorOpt = None,
    per_host: Annotated[
        int,
        typer.Option(
            "--per-host", min=1, help=f"Sandboxes packed per host VM (default {DEFAULT_SANDBOXES_PER_HOST})."
        ),
    ] = DEFAULT_SANDBOXES_PER_HOST,
    max_hosts: Annotated[
        int | None, typer.Option("--max-hosts", min=1, help="Optional cap on the number of host VMs.")
    ] = None,
    timeout: TimeoutOpt = None,
    idle_timeout: Annotated[
        str | None,
        typer.Option(help="Auto-terminate an idle host after this much inactivity (e.g. '10m'). Defaults to 10m."),
    ] = None,
    env: EnvOpt = None,
    secrets: SecretsOpt = None,
    env_file: EnvFileOpt = None,
    secrets_file: SecretsFileOpt = None,
    namespace: NamespaceOpt = None,
    forward_hf_token: Annotated[
        bool, typer.Option("--forward-hf-token", help="Inject your HF token as HF_TOKEN in each sandbox.")
    ] = False,
) -> None:
    """Define a sandbox pool. Starts no host and bills nothing — just saves the config.

    Returns a pool id; spawn sandboxes from it with `hf sandbox pool spawn <id>`, which
    inherits the image/flavor/timeout/env/secrets defined here.
    """
    pool_id = f"pool-{uuid.uuid4().hex[:12]}"
    config: dict[str, Any] = {
        "id": pool_id,
        "image": image,
        "flavor": flavor or "cpu-basic",
        "sandboxes_per_host": per_host,
        "max_hosts": max_hosts,
        "timeout": timeout,
        "idle_timeout": idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT,
        "env": parse_env_map(env, env_file),
        "secrets": parse_env_map(secrets, secrets_file),
        "forward_hf_token": forward_hf_token,
        "namespace": namespace,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_pool(config)
    out.result("Pool created", id=pool_id, image=config["image"], flavor=config["flavor"])
    out.hint(f"Spawn a sandbox with `hf sandbox pool spawn {pool_id}`.")
    out.hint(f"Delete the pool (and its hosts) with `hf sandbox pool delete {pool_id}`.")


@pool_cli.command(
    "spawn",
    examples=[
        "hf sandbox pool spawn <pool_id>",
        "hf sandbox pool spawn <pool_id> -n 100",
    ],
)
def pool_spawn(
    pool_id: Annotated[str, typer.Argument(help="Pool id from `hf sandbox pool create`.")],
    num: Annotated[int, typer.Option("-n", "--num", min=1, help="How many sandboxes to spawn.")] = 1,
    token: TokenOpt = None,
) -> None:
    """Spawn sandbox(es) from a pool, reusing a warm host or booting one as needed.

    A single spawn packs onto a warm host (found via job labels, possibly left by an
    earlier spawn or another machine) before booting a new one — so you can grow on
    demand one sandbox at a time. Hosts stop billing after the pool's idle timeout or
    via `hf sandbox kill`.
    """
    config = _load_pool(pool_id)
    start = time.time()
    pool = SandboxPool(
        image=config["image"],
        flavor=config["flavor"],
        sandboxes_per_host=config["sandboxes_per_host"],
        max_hosts=config.get("max_hosts"),
        name=pool_id,
        timeout=config.get("timeout"),
        idle_timeout=config.get("idle_timeout"),
        env=config.get("env") or None,
        secrets=config.get("secrets") or None,
        forward_hf_token=config.get("forward_hf_token", False),
        namespace=config.get("namespace"),
        token=token,
    )
    created = pool.create(count=num)
    sandboxes = created if isinstance(created, list) else [created]
    if len(sandboxes) == 1:
        sbx = sandboxes[0]
        out.result("Sandbox ready", id=sbx.id, host=sbx.host_id, pool=pool_id, elapsed=f"{time.time() - start:.1f}s")
    else:
        out.table([{"id": sbx.id} for sbx in sandboxes], id_key="id")
        out.result(
            "Sandboxes ready",
            count=len(sandboxes),
            hosts=len(pool.host_ids),
            pool=pool_id,
            elapsed=f"{time.time() - start:.1f}s",
        )
    out.hint(f"Run a command with `hf sandbox exec {sandboxes[0].id} -- echo hello`.")
    out.hint(f"Spawn another onto the same host(s) with `hf sandbox pool spawn {pool_id}`.")


@pool_cli.command("ls | list", examples=["hf sandbox pool ls"])
def pool_ls() -> None:
    """List locally-defined sandbox pools."""
    rows = []
    if _POOLS_DIR.is_dir():
        for path in sorted(_POOLS_DIR.glob("*.json")):
            try:
                c = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "id": c.get("id", path.stem),
                    "image": c.get("image", ""),
                    "flavor": c.get("flavor", ""),
                    "per_host": c.get("sandboxes_per_host", ""),
                    "created": c.get("created_at", ""),
                }
            )
    out.table(rows, id_key="id")
    if not rows:
        out.hint("Define one with `hf sandbox pool create`.")
    else:
        out.hint("Spawn a sandbox with `hf sandbox pool spawn <id>`, or see running ones with `hf sandbox ls`.")


@pool_cli.command(
    "delete | rm",
    examples=["hf sandbox pool delete <pool_id>"],
)
def pool_delete(
    pool_id: Annotated[str, typer.Argument(help="Pool id to delete.")],
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Answer Yes to prompts automatically.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Delete a pool definition and terminate any host VMs it has running."""
    config = _load_pool(pool_id)
    api = get_hf_api(token=token)
    ns = namespace or config.get("namespace")
    hosts = [
        job
        for job in api.list_jobs(namespace=ns)
        if (job.labels or {}).get(POOL_LABEL) == pool_id and job.status.stage == "RUNNING"
    ]
    suffix = f" and terminate {len(hosts)} running host(s)?" if hosts else "?"
    out.confirm(f"Delete pool '{pool_id}'{suffix}", yes=yes)
    for job in hosts:
        api.cancel_job(job_id=job.id, namespace=job.owner.name)
    _pool_path(pool_id).unlink(missing_ok=True)
    out.result("Pool deleted", id=pool_id, hosts_terminated=len(hosts))
