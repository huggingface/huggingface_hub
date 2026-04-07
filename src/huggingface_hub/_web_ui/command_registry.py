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
"""Registry of available CLI commands and their metadata for the web UI."""

import inspect
from typing import Annotated, Any, get_args, get_origin

import typer
from typer.models import ArgumentInfo, CommandInfo, DefaultPlaceholder, OptionInfo
from typer.models import Context as TyperContext


CommandEntry = dict[str, str | list[str]]
CommandMap = dict[str, list[CommandEntry]]


class CommandRegistry:
    """Registry of all available CLI commands with their parameters."""

    @staticmethod
    def _is_hidden(value: bool | DefaultPlaceholder[bool] | None) -> bool:
        if isinstance(value, DefaultPlaceholder):
            return bool(value.value)
        return bool(value)

    @staticmethod
    def _primary_name(raw_name: str) -> str:
        return raw_name.split("|")[0].strip()

    @classmethod
    def _get_param_metadata(cls, parameter: inspect.Parameter) -> ArgumentInfo | OptionInfo | None:
        annotation = parameter.annotation

        # Preferred style in the codebase: Annotated[..., ArgumentInfo/OptionInfo]
        if get_origin(annotation) is Annotated:
            metadata = get_args(annotation)[1:]
            for entry in metadata:
                if isinstance(entry, (ArgumentInfo, OptionInfo)):
                    return entry

        # Fallback for older Typer style where default itself can carry info
        if isinstance(parameter.default, (ArgumentInfo, OptionInfo)):
            return parameter.default

        return None

    @staticmethod
    def _is_context_parameter(parameter: inspect.Parameter) -> bool:
        annotation = parameter.annotation
        return annotation in {TyperContext, typer.Context}

    @classmethod
    def _extract_args_and_flags(cls, callback: Any) -> tuple[list[str], list[str]]:
        args: list[str] = []
        flags: list[str] = []

        for parameter in inspect.signature(callback).parameters.values():
            if cls._is_context_parameter(parameter):
                continue

            metadata = cls._get_param_metadata(parameter)

            if isinstance(metadata, ArgumentInfo):
                args.append(parameter.name)
                continue

            if isinstance(metadata, OptionInfo):
                declarations = [decl for decl in (metadata.param_decls or ()) if decl]
                option_names = [decl for decl in declarations if decl.startswith("-")]

                if option_names:
                    # Prefer long options to make the web form clearer.
                    long_options = [name for name in option_names if name.startswith("--")]
                    flags.append(long_options[0] if long_options else option_names[0])
                else:
                    flags.append(f"--{parameter.name.replace('_', '-')}")

            if metadata is None and parameter.default is inspect._empty:
                args.append(parameter.name)

        return args, flags

    @classmethod
    def _add_command(
        cls,
        target: CommandMap,
        command_info: CommandInfo,
        prefix: list[str],
        category: str,
    ) -> None:
        if cls._is_hidden(command_info.hidden):
            return

        callback = command_info.callback
        if callback is None:
            return

        raw_command_name = command_info.name or callback.__name__.replace("_", "-")
        command_name = cls._primary_name(raw_command_name)
        full_name = " ".join(prefix + [command_name]).strip()

        args, flags = cls._extract_args_and_flags(callback)
        callback_doc = inspect.getdoc(callback) or ""
        callback_summary = callback_doc.splitlines()[0] if callback_doc else ""
        description = command_info.help or callback_summary

        target.setdefault(category, []).append(
            {
                "name": full_name,
                "description": description or f"Run `{full_name}` command",
                "args": args,
                "flags": flags,
            }
        )

    @classmethod
    def _walk_typer(cls, app: typer.Typer, registry: CommandMap, prefix: list[str], category: str) -> None:
        for command_info in app.registered_commands:
            cls._add_command(registry, command_info, prefix=prefix, category=category)

        for group_info in app.registered_groups:
            if cls._is_hidden(group_info.hidden):
                continue

            group_name = cls._primary_name(group_info.name or "")
            if not group_name:
                continue

            subgroup = group_info.typer_instance
            subgroup_category = group_name.replace("-", " ").title()
            cls._walk_typer(subgroup, registry, prefix=prefix + [group_name], category=subgroup_category)

    @classmethod
    def from_typer_app(cls, app: typer.Typer) -> CommandMap:
        """Build command metadata from a Typer app instance."""
        registry: CommandMap = {}
        cls._walk_typer(app, registry, prefix=[], category="Main")

        # Keep ordering stable for deterministic UI and tests.
        for entries in registry.values():
            entries.sort(key=lambda entry: str(entry["name"]))
        return dict(sorted(registry.items(), key=lambda item: item[0]))

    @classmethod
    def get_all_commands(cls) -> dict[str, Any]:
        """Get all available commands organized by category from the live Typer app."""
        # Local import to avoid circular dependency at module import time.
        from huggingface_hub.cli.hf import app

        return cls.from_typer_app(app)

    @classmethod
    def get_command_by_category(cls, category: str) -> list[dict[str, Any]]:
        """Get commands in a specific category."""
        return cls.get_all_commands().get(category, [])

    @classmethod
    def get_all_command_names(cls) -> list[str]:
        """Get flat list of all command names."""
        all_commands = []
        for commands in cls.get_all_commands().values():
            all_commands.extend([str(cmd["name"]) for cmd in commands])
        return sorted(all_commands)
