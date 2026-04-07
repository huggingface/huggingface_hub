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
"""Tests for the web UI module."""

import asyncio

import pytest
import typer


class TestCommandRegistry:
    """Test the command registry."""

    def test_import_command_registry(self):
        """Test that CommandRegistry can be imported."""
        from huggingface_hub._web_ui.command_registry import CommandRegistry

        assert CommandRegistry is not None

    def test_get_all_commands(self):
        """Test getting all commands."""
        from huggingface_hub._web_ui.command_registry import CommandRegistry

        commands = CommandRegistry.get_all_commands()
        assert isinstance(commands, dict)
        assert len(commands) > 0
        assert "Main" in commands
        assert "Models" in commands

    def test_get_command_by_category(self):
        """Test getting commands by category from dynamic registry."""
        from huggingface_hub._web_ui.command_registry import CommandRegistry

        models_commands = CommandRegistry.get_command_by_category("Models")
        assert isinstance(models_commands, list)
        assert len(models_commands) > 0
        assert any(str(cmd["name"]).startswith("models ") for cmd in models_commands)

    def test_get_all_command_names(self):
        """Test getting all command names."""
        from huggingface_hub._web_ui.command_registry import CommandRegistry

        names = CommandRegistry.get_all_command_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "download" in names
        assert "upload" in names
        assert all("|" not in name for name in names)

    def test_from_typer_app_discovers_new_commands(self):
        """Test dynamic discovery on an arbitrary Typer app."""
        from huggingface_hub._web_ui.command_registry import CommandRegistry

        test_app = typer.Typer()
        plugins = typer.Typer()

        @test_app.command()
        def ping(target: str, count: int = typer.Option(1, "--count")) -> None:
            return None

        @plugins.command()
        def sync(repo_id: str) -> None:
            return None

        test_app.add_typer(plugins, name="plugins")

        registry = CommandRegistry.from_typer_app(test_app)
        names = [command["name"] for commands in registry.values() for command in commands]

        assert "ping" in names
        assert "plugins sync" in names

    def test_from_typer_app_ignores_context_parameters(self):
        """Test that Typer/Click context parameters are not exposed as user inputs."""
        from huggingface_hub._web_ui.command_registry import CommandRegistry

        test_app = typer.Typer()

        @test_app.command()
        def extension_exec(ctx: typer.Context) -> None:
            return None

        registry = CommandRegistry.from_typer_app(test_app)
        names = [command["name"] for commands in registry.values() for command in commands]

        assert "extension-exec" in names
        extension_exec_entry = next(command for command in registry["Main"] if command["name"] == "extension-exec")
        assert extension_exec_entry["args"] == []


class TestCommandExecutor:
    """Test the command executor."""

    def test_import_command_executor(self):
        """Test that CommandExecutor can be imported."""
        from huggingface_hub._web_ui.command_executor import CommandExecutor

        assert CommandExecutor is not None

    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        """Test executing a simple command."""

        from huggingface_hub._web_ui.command_executor import CommandExecutor

        # Test with echo command (cross-platform)
        output_lines = []
        async for output in CommandExecutor.execute_command(["echo", "test"]):
            output_lines.append(output)

        # Should have received some output
        assert len(output_lines) > 0

    @pytest.mark.asyncio
    async def test_execute_command_kills_process_on_close(self, monkeypatch):
        """Test that the subprocess is cleaned up when the generator closes early."""
        from huggingface_hub._web_ui.command_executor import CommandExecutor

        class FakeStdout:
            def __init__(self):
                self.reads = 0

            async def readline(self):
                self.reads += 1
                if self.reads == 1:
                    return b"hello\n"
                await asyncio.sleep(0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.stdout = FakeStdout()
                self.returncode = None
                self.killed = False
                self.wait_called = False

            def kill(self):
                self.killed = True
                self.returncode = -9

            async def wait(self):
                self.wait_called = True
                return self.returncode

        fake_process = FakeProcess()

        async def fake_create_subprocess_exec(*args, **kwargs):
            return fake_process

        monkeypatch.setattr("huggingface_hub._web_ui.command_executor.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

        generator = CommandExecutor.execute_command(["echo", "test"])
        first_output = await anext(generator)
        assert first_output == "hello\n"

        await generator.aclose()

        assert fake_process.killed is True
        assert fake_process.wait_called is True


class TestServer:
    """Test the FastAPI server."""

    def test_create_app(self):
        """Test that the app can be created."""
        from huggingface_hub._web_ui.server import create_app

        app = create_app()
        assert app is not None

    def test_create_app_uses_custom_startup_url(self, capsys):
        """Test that the startup banner reflects the configured host and port."""
        from fastapi.testclient import TestClient

        from huggingface_hub._web_ui.server import create_app

        app = create_app(host="127.0.0.1", port=9000)

        with TestClient(app):
            pass

        captured = capsys.readouterr()
        assert "Open your browser and go to: http://127.0.0.1:9000" in captured.out

    def test_app_has_routes(self):
        """Test that the app has expected routes."""
        from huggingface_hub._web_ui.server import create_app

        app = create_app()
        routes = [route.path for route in app.routes]

        assert "/api/health" in routes
        assert "/api/commands" in routes
        assert "/ws/execute" in routes

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test the health check endpoint."""
        from fastapi.testclient import TestClient

        from huggingface_hub._web_ui.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_commands_endpoint(self):
        """Test the commands endpoint."""
        from fastapi.testclient import TestClient

        from huggingface_hub._web_ui.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/commands")
        assert response.status_code == 200
        data = response.json()
        assert "commands" in data
        assert isinstance(data["commands"], dict)

    def test_resolve_frontend_static_path_allows_safe_paths(self, tmp_path):
        """Test that safe frontend-relative paths resolve within the root."""
        from huggingface_hub._web_ui.server import _resolve_frontend_static_path

        frontend_root = tmp_path / "frontend"
        frontend_root.mkdir()
        asset_path = frontend_root / "app.js"
        asset_path.write_text("console.log('ok');", encoding="utf-8")

        resolved = _resolve_frontend_static_path("app.js", frontend_root)

        assert resolved == asset_path.resolve()

    def test_resolve_frontend_static_path_rejects_absolute_path(self, tmp_path):
        """Test that absolute paths are rejected."""
        from huggingface_hub._web_ui.server import _resolve_frontend_static_path

        frontend_root = tmp_path / "frontend"
        frontend_root.mkdir()
        forbidden = frontend_root.parent / "secret.txt"
        forbidden.write_text("nope", encoding="utf-8")

        resolved = _resolve_frontend_static_path(str(forbidden), frontend_root)

        assert resolved is None

    def test_resolve_frontend_static_path_rejects_traversal(self, tmp_path):
        """Test that traversal outside the root is rejected."""
        from huggingface_hub._web_ui.server import _resolve_frontend_static_path

        frontend_root = tmp_path / "frontend"
        frontend_root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("nope", encoding="utf-8")

        resolved = _resolve_frontend_static_path("../secret.txt", frontend_root)

        assert resolved is None

    def test_websocket_accepts_same_origin(self, monkeypatch):
        """Test that the websocket accepts same-origin connections."""
        from fastapi.testclient import TestClient

        from huggingface_hub._web_ui import server as web_server

        async def fake_execute_command(command):
            yield "hello from websocket\n"
            yield web_server.EXIT_CODE_PREFIX + "0"

        monkeypatch.setattr(web_server.CommandExecutor, "execute_command", fake_execute_command)

        app = web_server.create_app()
        client = TestClient(app)

        with client.websocket_connect("/ws/execute", headers={"origin": "http://testserver"}) as websocket:
            websocket.send_json({"command": ["--version"]})

            assert websocket.receive_json() == {"output": "hello from websocket\n"}
            assert websocket.receive_json() == {"status": "completed"}

    def test_websocket_rejects_cross_origin(self):
        """Test that the websocket rejects cross-site origins."""
        from fastapi.testclient import TestClient

        from huggingface_hub._web_ui.server import create_app

        app = create_app()
        client = TestClient(app)

        with pytest.raises(Exception):
            with client.websocket_connect("/ws/execute", headers={"origin": "http://evil.example"}):
                pass


class TestWebUICLI:
    """Test the web UI CLI command."""

    def test_import_web_ui_cli(self):
        """Test that web_ui_cli can be imported."""
        from huggingface_hub.cli.web_ui import web_ui_cli

        assert web_ui_cli is not None

    def test_web_ui_command_registered(self):
        """Test that the web-ui command is registered in the main CLI."""
        from typer.testing import CliRunner

        from huggingface_hub.cli.hf import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "web-ui" in result.stdout

    def test_browser_url_normalization(self):
        """Test that wildcard hosts map to localhost browser URLs."""
        from huggingface_hub.cli.web_ui import _browser_url

        assert _browser_url("0.0.0.0", 7860) == "http://127.0.0.1:7860"
        assert _browser_url("127.0.0.1", 7860) == "http://127.0.0.1:7860"

    def test_open_browser_waits_for_server(self, monkeypatch):
        """Test that the browser is only opened after the health endpoint responds."""
        import types

        from huggingface_hub.cli import web_ui

        responses = [
            types.SimpleNamespace(status_code=503),
            types.SimpleNamespace(status_code=200),
        ]
        opened_urls = []
        sleeps = []

        monkeypatch.setattr(web_ui.httpx, "get", lambda *args, **kwargs: responses.pop(0))
        monkeypatch.setattr(web_ui.webbrowser, "open", lambda url, new=2: opened_urls.append(url))
        monkeypatch.setattr(web_ui, "sleep", lambda seconds: sleeps.append(seconds))

        web_ui._open_browser_when_ready("127.0.0.1", 7860)

        assert opened_urls == ["http://127.0.0.1:7860"]
        assert sleeps == [0.2]

    def test_open_browser_timeout(self, monkeypatch):
        """Test that the browser is not opened if the server never becomes ready."""
        from huggingface_hub.cli import web_ui

        opened_urls = []

        monkeypatch.setattr(web_ui, "_wait_for_server_ready", lambda port: False)
        monkeypatch.setattr(web_ui.webbrowser, "open", lambda url, new=2: opened_urls.append(url))

        web_ui._open_browser_when_ready("127.0.0.1", 7860)

        assert opened_urls == []
