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
"""Unit tests for the minimal Click framework (`huggingface_hub.cli._framework`).

These exercise the framework in isolation (no `hf` commands), covering the
annotation -> click.Parameter conversion, value handling at invocation time,
help-record rendering, and the group/command decorator API.
"""

import enum
from pathlib import Path
from typing import Annotated, Literal

import click
import pytest
from click.testing import CliRunner

from huggingface_hub.cli._framework import (
    Argument,
    HfGroup,
    Option,
    build_command,
    get_command_name,
)


class Color(str, enum.Enum):
    red = "red"
    blue = "blue"


def _params(func) -> dict:
    return {p.name: p for p in build_command(func, name="x").params}


def test_get_command_name_kebab_cases():
    assert get_command_name("force_download") == "force-download"
    assert get_command_name("type") == "type"


class TestParameterBuilding:
    def test_required_argument(self):
        def f(repo_id: Annotated[str, Argument()]): ...

        param = _params(f)["repo_id"]
        assert isinstance(param, click.Argument) and param.required and param.type is click.STRING

    def test_optional_list_argument_uses_nargs_minus_one(self):
        def f(files: Annotated[list[str] | None, Argument()] = None): ...

        param = _params(f)["files"]
        assert isinstance(param, click.Argument) and not param.required and param.nargs == -1

    def test_bool_option_without_decl_generates_toggle(self):
        def f(force: Annotated[bool, Option()] = False): ...

        param = _params(f)["force"]
        assert param.is_flag and param.opts == ["--force"] and param.secondary_opts == ["--no-force"]

    def test_bool_option_with_explicit_decl_is_single_flag(self):
        def f(meta: Annotated[bool, Option("--meta")] = False): ...

        param = _params(f)["meta"]
        assert param.is_flag and param.opts == ["--meta"] and not param.secondary_opts

    def test_optional_str_option_defaults_none(self):
        def f(revision: Annotated[str | None, Option()] = None): ...

        param = _params(f)["revision"]
        assert isinstance(param, click.Option) and not param.required and param.default is None

    def test_int_option_with_min_is_intrange(self):
        def f(workers: Annotated[int, Option(min=1)] = 1): ...

        param = _params(f)["workers"]
        assert isinstance(param.type, click.IntRange) and param.type.min == 1

    def test_enum_option_builds_choice_from_values(self):
        def f(color: Annotated[Color, Option("--color")] = Color.red): ...

        param = _params(f)["color"]
        assert isinstance(param.type, click.Choice) and list(param.type.choices) == ["red", "blue"]

    def test_literal_option_builds_choice(self):
        def f(region: Annotated[Literal["us", "eu"], Option("--region")] = "us"): ...

        param = _params(f)["region"]
        assert isinstance(param.type, click.Choice) and list(param.type.choices) == ["us", "eu"]

    def test_path_option(self):
        def f(path: Annotated[Path | None, Option()] = None): ...

        assert isinstance(_params(f)["path"].type, click.Path)

    def test_multi_decl_option_name_matches_python_param(self):
        # The Python name is prepended as the first decl so the handler kwarg lines up
        # regardless of the user-facing flag names.
        def f(volumes: Annotated[list[str] | None, Option("-v", "--volume")] = None): ...

        param = _params(f)["volumes"]
        assert param.name == "volumes" and "-v" in param.opts and "--volume" in param.opts and param.multiple

    def test_explicit_click_type_overrides_inference(self):
        soft_choice = click.Choice(["a", "b"])

        def f(mode: Annotated[str, Option("--mode", click_type=soft_choice)] = "a"): ...

        assert _params(f)["mode"].type is soft_choice

    def test_unsupported_annotation_raises(self):
        def f(bad: Annotated[dict, Option("--bad")] = None): ...

        with pytest.raises(TypeError):
            build_command(f, name="x")


class TestInvocation:
    def test_enum_flag_and_list_reach_handler(self):
        captured = {}

        def f(
            repo_id: Annotated[str, Argument()],
            color: Annotated[Color, Option("--color")] = Color.red,
            force: Annotated[bool, Option()] = False,
            volumes: Annotated[list[str] | None, Option("-v", "--volume")] = None,
        ):
            captured.update(repo_id=repo_id, color=color, force=force, volumes=volumes)

        result = CliRunner().invoke(
            build_command(f, name="run"), ["r", "--color", "blue", "--force", "-v", "a", "-v", "b"]
        )
        assert result.exit_code == 0, result.output
        assert captured == {"repo_id": "r", "color": Color.blue, "force": True, "volumes": ["a", "b"]}

    def test_value_callback_is_applied(self):
        seen = {}

        def f(name: Annotated[str | None, Option("--name", callback=lambda v: (v or "").upper())] = None):
            seen["name"] = name

        CliRunner().invoke(build_command(f, name="c"), ["--name", "hi"])
        assert seen["name"] == "HI"

    def test_context_is_injected(self):
        seen = {}

        def f(ctx: click.Context, value: Annotated[str | None, Option("--value")] = None):
            seen["is_ctx"] = isinstance(ctx, click.Context)

        CliRunner().invoke(build_command(f, name="c"), ["--value", "1"])
        assert seen.get("is_ctx")


class TestHelpRecords:
    def _help(self, func) -> str:
        return CliRunner().invoke(build_command(func, name="x"), ["--help"]).output

    def test_argument_renders_help_and_required(self):
        def f(repo_id: Annotated[str, Argument(help="the repo")]): ...

        out = self._help(f)
        assert "REPO_ID" in out and "the repo" in out and "[required]" in out

    def test_bool_flag_default_string(self):
        def f(force: Annotated[bool, Option(help="force it")] = False): ...

        out = self._help(f)
        assert "--force / --no-force" in out and "[default: no-force]" in out

    def test_int_option_shows_range_but_argument_does_not(self):
        def f_opt(n: Annotated[int, Option(min=1, help="num")] = 1): ...

        def f_arg(n: Annotated[int, Argument(min=1, help="num")]): ...

        assert "x>=1" in self._help(f_opt)
        assert "x>=1" not in self._help(f_arg)


class TestGroup:
    def test_command_registration_preserves_order_and_aliases(self):
        group = HfGroup(name="root")

        @group.command("list | ls")
        def _list(): ...

        @group.command("info")
        def _info(): ...

        assert "list | ls" in group.commands
        assert group.list_commands(click.Context(group)) == ["list | ls", "info"]

    def test_add_group_sets_name_and_hidden(self):
        root = HfGroup(name="root")
        sub = HfGroup()
        root.add_group(sub, name="repos | repo", hidden=True)
        assert root.commands["repos | repo"] is sub and sub.name == "repos | repo" and sub.hidden

    def test_group_callback_sets_invoke_without_command_and_metavar(self):
        group = HfGroup(name="root")

        @group.group_callback(invoke_without_command=True)
        def _callback(): ...

        assert group.invoke_without_command and group.subcommand_metavar == "[COMMAND] [ARGS]..."
