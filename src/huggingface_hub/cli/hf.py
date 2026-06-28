# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import sys
import traceback
import importlib
from typing import Annotated, Any, Callable, Optional
import typer

from huggingface_hub import __version__, constants
from huggingface_hub.cli._cli_utils import check_cli_update, fallback_typer_group_factory, typer_factory
from huggingface_hub.cli._errors import format_known_exception
from huggingface_hub.cli._output import out
from huggingface_hub.utils import logging


# ==========================================
# 1. LAZY IMPORT UTILITIES
# ==========================================
def _lazy_reflect(module_path: str, attribute: str) -> Callable[..., Any]:
    """
    Dynamically import and return a module attribute at runtime.
    
    This reduces CLI startup time by deferring module imports until needed,
    and helps avoid circular import issues.
    
    Args:
        module_path: Full module path (e.g., "huggingface_hub.cli.auth")
        attribute: Attribute name to retrieve (e.g., "auth_cli")
    
    Returns:
        A proxy function that loads the attribute on first call
    """
    def proxy(*args, **kwargs):
        try:
            mod = importlib.import_module(module_path)
            func = getattr(mod, attribute)
            return func(*args, **kwargs)
        except ImportError as e:
            out.error(f"Failed to load command from {module_path}: {e}")
            sys.exit(1)
        except AttributeError:
            out.error(f"'{attribute}' not found in {module_path}")
            sys.exit(1)
    return proxy


def _lazy_import_examples(module_path: str, attr_name: str) -> Optional[dict]:
    """
    Lazily load CLI examples metadata.
    
    Args:
        module_path: Full module path
        attr_name: Attribute name for examples
    
    Returns:
        Examples dict or None if not found
    """
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name, None)
    except ImportError:
        return None


# Lazy wrappers for extensions
def _dispatch_unknown_extension_wrapper(*args, **kwargs):
    from huggingface_hub.cli.extensions import dispatch_unknown_top_level_extension
    return dispatch_unknown_top_level_extension(*args, **kwargs)


def _list_extensions_wrapper(*args, **kwargs):
    from huggingface_hub.cli.extensions import list_installed_extensions_for_help
    return list_installed_extensions_for_help(*args, **kwargs)


# Initialize Typer App
app = typer_factory(
    help="Hugging Face Hub CLI",
    cls=fallback_typer_group_factory(
        _dispatch_unknown_extension_wrapper,
        extra_commands_provider=_list_extensions_wrapper,
    ),
)


# ==========================================
# 2. CORE CALLBACKS & VERSION
# ==========================================
def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def app_callback(
    version: Annotated[
        bool | None, typer.Option("-v", "--version", callback=_version_callback, is_eager=True, hidden=True)
    ] = None,
) -> None:
    pass


# ==========================================
# 3. TOP-LEVEL COMMANDS (LAZILY LOADED)
# ==========================================
# Load examples lazily
cp_examples = _lazy_import_examples("huggingface_hub.cli._cp", "CP_EXAMPLES")
download_examples = _lazy_import_examples("huggingface_hub.cli.download", "DOWNLOAD_EXAMPLES")
upload_examples = _lazy_import_examples("huggingface_hub.cli.upload", "UPLOAD_EXAMPLES")
upload_large_folder_examples = _lazy_import_examples("huggingface_hub.cli.upload_large_folder", "UPLOAD_LARGE_FOLDER_EXAMPLES")

# Register commands with lazy loading
app.command(examples=cp_examples)(_lazy_reflect("huggingface_hub.cli._cp", "make_cp"))
app.command()(_lazy_reflect("huggingface_hub.cli.buckets", "sync"))
app.command(examples=download_examples)(_lazy_reflect("huggingface_hub.cli.download", "download"))
app.command(examples=upload_examples)(_lazy_reflect("huggingface_hub.cli.upload", "upload"))
app.command(examples=upload_large_folder_examples)(_lazy_reflect("huggingface_hub.cli.upload_large_folder", "upload_large_folder"))

# System commands
app.command(topic="help")(_lazy_reflect("huggingface_hub.cli.system", "env"))
app.command(topic="help")(_lazy_reflect("huggingface_hub.cli.system", "update"))
app.command(topic="help")(_lazy_reflect("huggingface_hub.cli.system", "version"))

# LFS internals (hidden)
app.command(hidden=True)(_lazy_reflect("huggingface_hub.cli.lfs", "lfs_enable_largefiles"))
app.command(hidden=True)(_lazy_reflect("huggingface_hub.cli.lfs", "lfs_multipart_upload"))


# ==========================================
# 4. COMMAND GROUPS (DYNAMIC REGISTRATION)
# ==========================================
def _register_subcommands() -> None:
    """
    Register all CLI subcommand groups with graceful error handling.
    
    If a subcommand fails to load, it will be skipped with a warning
    rather than crashing the entire CLI.
    """
    subcommands = [
        ("auth", "huggingface_hub.cli.auth", "auth_cli", {}),
        ("buckets", "huggingface_hub.cli.buckets", "buckets_cli", {}),
        ("cache", "huggingface_hub.cli.cache", "cache_cli", {}),
        ("collections", "huggingface_hub.cli.collections", "collections_cli", {}),
        ("datasets", "huggingface_hub.cli.datasets", "datasets_cli", {}),
        ("discussions", "huggingface_hub.cli.discussions", "discussions_cli", {}),
        ("jobs", "huggingface_hub.cli.jobs", "jobs_cli", {}),
        ("models", "huggingface_hub.cli.models", "models_cli", {}),
        ("papers", "huggingface_hub.cli.papers", "papers_cli", {}),
        ("repos | repo", "huggingface_hub.cli.repos", "repos_cli", {}),
        ("repo-files", "huggingface_hub.cli.repo_files", "repo_files_cli", {"hidden": True}),
        ("skills", "huggingface_hub.cli.skills", "skills_cli", {}),
        ("spaces", "huggingface_hub.cli.spaces", "spaces_cli", {}),
        ("webhooks", "huggingface_hub.cli.webhooks", "webhooks_cli", {}),
        ("endpoints", "huggingface_hub.cli.inference_endpoints", "ie_cli", {}),
        ("extensions | ext", "huggingface_hub.cli.extensions", "extensions_cli", {}),
    ]

    for name, module_path, attr, kwargs in subcommands:
        try:
            mod = importlib.import_module(module_path)
            cli = getattr(mod, attr)
            app.add_typer(cli, name=name, **kwargs)
        except (ImportError, AttributeError) as e:
            out.warning(f"Failed to load subcommand group '{name}': {e}")


_register_subcommands()


# ==========================================
# 5. MAIN ENTRY POINT WITH ERROR HANDLING
# ==========================================
def main() -> None:
    """
    Main entry point for the Hugging Face Hub CLI.
    
    Handles logging configuration, update checks, and comprehensive error handling.
    """
    # Configure logging
    if not constants.HF_DEBUG:
        logging.set_verbosity_info()

    # Check for CLI updates (non-blocking)
    try:
        check_cli_update("huggingface_hub")
    except Exception:
        # Silently skip update check on network failures or other issues
        # Only log if in debug mode
        if constants.HF_DEBUG:
            out.debug("Update check failed (continuing anyway)")

    # Execute CLI
    try:
        app()
    except typer.Exit:
        sys.exit(0)
    except (typer.Abort, KeyboardInterrupt):
        out.error("Operation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        message = format_known_exception(e)
        if message:
            out.error(message)
        else:
            out.error(f"An unexpected error occurred: {e}")

        if constants.HF_DEBUG:
            out.error("\n--- Full Traceback ---")
            traceback.print_exc()
        else:
            out.hint("Set HF_DEBUG=1 environment variable for full traceback.")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
