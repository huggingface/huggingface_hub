"""
Utility functions for formatting results from mcp.CallToolResult.

Taken from the JS SDK: https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/ResultFormatter.ts.
"""

from typing import List

from mcp import types as mcp_types


def format_result(result: mcp_types.CallToolResult) -> str:
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
