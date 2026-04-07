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
from typing import Annotated

import typer

from huggingface_hub._web_ui.server import create_app


web_ui_cli = typer.Typer(help="Launch web-based UI for Hugging Face Hub commands")


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

        app = create_app()

        # Try to open browser if requested
        if open_browser:
            try:
                url = f"http://{host}:{port}"
                typer.echo(f"\nOpening browser at {url}...", err=False)
                webbrowser.open(url, new=2)
            except Exception as e:
                typer.echo(f"Warning: could not open browser automatically: {e}", err=True)

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
