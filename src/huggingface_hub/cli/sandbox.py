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

Usage:
    # Create a sandbox (prints its id)
    hf sandbox create [IMAGE]

    # Run a command inside it (streams output, propagates exit code)
    hf sandbox exec <sandbox_id> -- python -c "print('hi')"

    # Copy files in and out (docker-style)
    hf sandbox cp local.txt <sandbox_id>:/tmp/remote.txt
    hf sandbox cp <sandbox_id>:/tmp/remote.txt local.txt

    # List / inspect / terminate
    hf sandbox ls
    hf sandbox ps <sandbox_id>
    hf sandbox url <sandbox_id> <port>
    hf sandbox kill <sandbox_id>
"""

import sys
import time
from typing import Annotated

import typer

from huggingface_hub._sandbox import DEFAULT_IDLE_TIMEOUT, DEFAULT_IMAGE, Sandbox
from huggingface_hub.errors import CLIError, SandboxError

from ._cli_utils import (
    EnvFileOpt,
    EnvOpt,
    SecretsFileOpt,
    SecretsOpt,
    TokenOpt,
    VolumesOpt,
    parse_env_map,
    parse_volumes,
    typer_factory,
)
from ._output import out
from .jobs import ExposeOpt, FlavorOpt, NamespaceOpt, TimeoutOpt


sandbox_cli = typer_factory(help="Run and manage sandboxes on Hugging Face Jobs.")

SandboxIdArg = Annotated[str, typer.Argument(help="The sandbox id (as printed by `hf sandbox create`).")]


@sandbox_cli.command(
    "create",
    examples=[
        "hf sandbox create",
        "hf sandbox create ubuntu:24.04",
        "hf sandbox create --flavor a10g-small --timeout 1h",
        "hf sandbox create --expose 8080 -e DEBUG=1",
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
    """Create a sandbox and wait until it is ready."""
    start = time.time()
    sandbox = Sandbox.create(
        image=image,
        flavor=flavor or "cpu-basic",
        timeout=timeout,
        idle_timeout=idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT,
        env=parse_env_map(env, env_file),
        secrets=parse_env_map(secrets, secrets_file),
        volumes=parse_volumes(volume),
        expose=expose,
        namespace=namespace,
        forward_hf_token=forward_hf_token,
        token=token,
    )
    out.result("Sandbox ready", id=sandbox.id, image=image, elapsed=f"{time.time() - start:.1f}s")
    out.hint(f"Run a command with `hf sandbox exec {sandbox.id} -- echo hello`.")
    out.hint(f"Terminate it with `hf sandbox kill {sandbox.id}`.")


@sandbox_cli.command("ls | list", examples=["hf sandbox ls"])
def sandbox_ls(
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """List your running sandboxes."""
    rows = [
        {
            "id": job.id,
            "image": job.docker_image or job.space_id,
            "flavor": job.flavor,
            "created": job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else "N/A",
        }
        for job in Sandbox.list(namespace=namespace, token=token)
    ]
    out.table(rows, id_key="id")
    if not rows:
        out.hint("Create one with `hf sandbox create`.")


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

    sandbox = Sandbox.connect(sandbox_id, namespace=namespace, token=token)
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
    sandbox = Sandbox.connect(sandbox_id, namespace=namespace, token=token)
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
        if ":" in ref and not ref.startswith((".", "/", "~")):
            sandbox_id, path = ref.split(":", 1)
            return sandbox_id, path
        return None, ref

    src_sandbox, src_path = parse(src)
    dst_sandbox, dst_path = parse(dst)
    if (src_sandbox is None) == (dst_sandbox is None):
        raise CLIError("Exactly one of SRC and DST must be a sandbox path (<sandbox_id>:<path>).")
    if src_sandbox is not None:
        sandbox = Sandbox.connect(src_sandbox, namespace=namespace, token=token)
        sandbox.files.download(src_path, dst_path)
    else:
        assert dst_sandbox is not None
        sandbox = Sandbox.connect(dst_sandbox, namespace=namespace, token=token)
        sandbox.files.upload(src_path, dst_path)
    out.result("Copied", src=src, dst=dst)


@sandbox_cli.command("url", examples=["hf sandbox url <sandbox_id> 8080"])
def sandbox_url(
    sandbox_id: SandboxIdArg,
    port: Annotated[int, typer.Argument(help="Container port (must have been exposed at creation).")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Print the public URL of an exposed sandbox port."""
    sandbox = Sandbox.connect(sandbox_id, namespace=namespace, token=token)
    out.text(sandbox.url(port))
    out.hint("Requests must include an HF token: `curl -H 'Authorization: Bearer ...' <url>`.")


@sandbox_cli.command("kill", examples=["hf sandbox kill <sandbox_id>"])
def sandbox_kill(
    sandbox_id: SandboxIdArg,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Terminate a sandbox."""
    try:
        sandbox = Sandbox.connect(sandbox_id, namespace=namespace, token=token)
    except SandboxError as e:
        raise CLIError(str(e)) from e
    sandbox.kill()
    out.result("Sandbox terminated", id=sandbox_id)
