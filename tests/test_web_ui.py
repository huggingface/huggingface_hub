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


class TestServer:
    """Test the FastAPI server."""

    def test_create_app(self):
        """Test that the app can be created."""
        from huggingface_hub._web_ui.server import create_app

        app = create_app()
        assert app is not None

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
