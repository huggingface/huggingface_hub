"""
Utility functions for MCPClient and Tiny Agents.

Formatting utilities taken from the JS SDK: https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/ResultFormatter.ts.
"""

import importlib.resources as importlib_resources
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .constants import DEFAULT_AGENT, FILENAME_CONFIG, FILENAME_PROMPT


if TYPE_CHECKING:
    from mcp import types as mcp_types


def format_result(result: "mcp_types.CallToolResult") -> str:
    """
    Formats a mcp.types.CallToolResult content into a human-readable string.

    Args:
        result (CallToolResult)
            Object returned by mcp.ClientSession.call_tool.

    Returns:
        str
            A formatted string representing the content of the result.
    """
    content = result.content

    if len(content) == 0:
        return "[No content]"

    formatted_parts: List[str] = []

    for item in content:
        if item.type == "text":
            formatted_parts.append(item.text)

        elif item.type == "image":
            formatted_parts.append(
                f"[Binary Content: Image {item.mimeType}, {_get_base64_size(item.data)} bytes]\n"
                f"The task is complete and the content accessible to the User"
            )

        elif item.type == "audio":
            formatted_parts.append(
                f"[Binary Content: Audio {item.mimeType}, {_get_base64_size(item.data)} bytes]\n"
                f"The task is complete and the content accessible to the User"
            )

        elif item.type == "resource":
            resource = item.resource

            if hasattr(resource, "text"):
                formatted_parts.append(resource.text)

            elif hasattr(resource, "blob"):
                formatted_parts.append(
                    f"[Binary Content ({resource.uri}): {resource.mimeType}, {_get_base64_size(resource.blob)} bytes]\n"
                    f"The task is complete and the content accessible to the User"
                )

    return "\n".join(formatted_parts)


def _get_base64_size(base64_str: str) -> int:
    """Estimate the byte size of a base64-encoded string."""
    # Remove any prefix like "data:image/png;base64,"
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    padding = 0
    if base64_str.endswith("=="):
        padding = 2
    elif base64_str.endswith("="):
        padding = 1

    return (len(base64_str) * 3) // 4 - padding


def _load_builtin_agent_path(name: str) -> Optional[Path]:
    try:
        base = importlib_resources.files("huggingface_hub.inference._mcp.agents")
    except (ModuleNotFoundError, AttributeError):
        return None
    candidate_traversable = base / name
    candidate_path = Path(str(candidate_traversable))
    return candidate_path if candidate_path.is_dir() else None


def _load_config(source: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load server config and prompt."""
    if source is None:
        return (
            DEFAULT_AGENT,
            None,
        )

    path = Path(source).expanduser()

    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), None

    if path.is_dir():
        config_json = (path / FILENAME_CONFIG).read_text(encoding="utf-8")
        try:
            prompt_md = (path / FILENAME_PROMPT).read_text(encoding="utf-8")
        except FileNotFoundError:
            prompt_md = None
        return json.loads(config_json), prompt_md

    builtin_dir = _load_builtin_agent_path(source)
    if builtin_dir is not None:
        config_json = (builtin_dir / FILENAME_CONFIG).read_text(encoding="utf-8")
        try:
            prompt_md = (builtin_dir / FILENAME_PROMPT).read_text(encoding="utf-8")
        except FileNotFoundError:
            prompt_md = None
        return json.loads(config_json), prompt_md

    raise FileNotFoundError(source)


def _url_to_server_config(url: str, hf_token: Optional[str]) -> Dict:
    return {
        "command": None,
        "url": url,
        "env": {"AUTHORIZATION": f"Bearer {hf_token}"} if hf_token else None,
    }
