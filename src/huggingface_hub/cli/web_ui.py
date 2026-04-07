# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Web UI CLI command for the Hugging Face Hub."""

import sys
import webbrowser
from threading import Thread
from time import monotonic, sleep
from typing import Annotated

import httpx
import typer

from huggingface_hub._web_ui.server import create_app


web_ui_cli = typer.Typer(help="Launch web-based UI for Hugging Face Hub commands")


def _browser_url(host: str, port: int) -> str:
    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        browser_host = "127.0.0.1"
    else:
        browser_host = host
    return f"http://{browser_host}:{port}"


def _wait_for_server_ready(port: int, timeout_seconds: int = 30) -> bool:
    health_url = f"http://127.0.0.1:{port}/api/health"
    deadline = timeout_seconds + monotonic()

    while monotonic() < deadline:
        try:
            response = httpx.get(health_url, timeout=1.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        sleep(0.2)

    return False


def _open_browser_when_ready(host: str, port: int) -> None:
    if not _wait_for_server_ready(port):
        typer.echo("Warning: web UI did not become ready before the browser launcher timed out.", err=True)
        return

    url = _browser_url(host, port)
    try:
        typer.echo(f"\nOpening browser at {url}...", err=False)
        webbrowser.open(url, new=2)
    except Exception as e:
        typer.echo(f"Warning: could not open browser automatically: {e}", err=True)


@web_ui_cli.command()
def run(
    host: Annotated[str, typer.Option("--host", help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind to")] = 7860,
    open_browser: Annotated[
        bool, typer.Option("--open-browser/--no-open-browser", help="Automatically open browser")
    ] = True,
) -> None:
    """
    Launch the web UI for executing Hugging Face Hub commands.

    The web UI provides a form-based interface to run any CLI command available in the `hf` CLI,
    with real-time output streaming.

    Example:
        hf web-ui run
        hf web-ui run --host 0.0.0.0 --port 8000
    """
    try:
        try:
            import uvicorn
        except ImportError as e:
            typer.echo(
                "Error: missing dependency 'uvicorn'. Install with `pip install 'huggingface_hub[oauth]'`.",
                err=True,
            )
            raise typer.Exit(code=1) from e

        app = create_app(host=host, port=port)

        # Open the browser only after the health endpoint is reachable.
        if open_browser:
            Thread(target=_open_browser_when_ready, args=(host, port), daemon=True).start()

        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )

    except KeyboardInterrupt:
        typer.echo("\n\nWeb UI stopped.", err=False)
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Error starting web UI: {e}", err=True)
        sys.exit(1)
