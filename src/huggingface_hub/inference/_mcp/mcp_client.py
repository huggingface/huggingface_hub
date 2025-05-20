import json
import logging
import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, AsyncIterable, Dict, List, Optional, Union

from typing_extensions import TypeAlias

from .._generated._async_client import AsyncInferenceClient
from .._generated.types import (
    ChatCompletionInputMessage,
    ChatCompletionInputTool,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputDeltaToolCall,
)
from .._providers import PROVIDER_OR_POLICY_T
from .utils import ServerConfigTypes, format_result


if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)

# Type alias for tool names
ToolName: TypeAlias = str


class MCPClient:
    """
    Client for connecting to one or more MCP servers and processing chat completions with tools.

    <Tip warning={true}>

    This class is experimental and might be subject to breaking changes in the future without prior notice.

    </Tip>
    """

    def __init__(
        self,
        *,
        model: str,
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        api_key: Optional[str] = None,
    ):
        # Initialize MCP sessions as a dictionary of ClientSession objects
        self.sessions: Dict[ToolName, "ClientSession"] = {}
        self.exit_stack = AsyncExitStack()
        self.available_tools: List[ChatCompletionInputTool] = []

        # Initialize the AsyncInferenceClient
        self.client = AsyncInferenceClient(model=model, provider=provider, api_key=api_key)

    async def __aenter__(self):
        """Enter the context manager"""
        await self.client.__aenter__()
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager"""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
        await self.cleanup()

    async def add_mcp_server(
        self,
        server: Union[ServerConfigTypes, Dict],
    ):
        """Connect to an MCP server"""
        from mcp import ClientSession
        from mcp import types as mcp_types
        from mcp.client.sse import sse_client
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from yarl import URL

        def _merge_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
            merged = dict(env or {})
            merged.setdefault("PATH", os.getenv("PATH", ""))
            return merged

        if "type" not in server:
            params = StdioServerParameters(
                command=server["command"],
                args=server.get("args", []),
                env=_merge_env(server.get("env")),
                cwd=server.get("cwd"),
            )
            read, write = await self.exit_stack.enter_async_context(stdio_client(params))
        else:
            cfg = server["config"]
            if server["type"] == "stdio":
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=_merge_env(cfg.get("env")),
                    cwd=cfg.get("cwd"),
                )
                read, write = await self.exit_stack.enter_async_context(stdio_client(params))
            elif server["type"] == "sse":
                url = URL(cfg["url"])
                read, write = await self.exit_stack.enter_async_context(sse_client(url, cfg.get("options")))
            else:
                raise ValueError(f"Unsupported server type: {server['type']}")

        # Connect MCP session
        session = await self.exit_stack.enter_async_context(
            ClientSession(
                read_stream=read,
                write_stream=write,
                client_info=mcp_types.Implementation(
                    name="huggingface_hub.MCPClient",
                    version="0.1",
                ),
            )
        )

        await session.initialize()
        tools_result = await session.list_tools()
        logger.debug("Connected to server with tools: %s", [tool.name for tool in tools_result.tools])

        for tool in tools_result.tools:
            if tool.name in self.sessions:
                logger.warning(f"Tool '{tool.name}' already defined by another server. Skipping.")
                continue
            self.sessions[tool.name] = session
            self.available_tools.append(
                ChatCompletionInputTool.parse_obj_as_instance(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
            )

    async def process_single_turn_with_tools(
        self,
        messages: List[Union[Dict, ChatCompletionInputMessage]],
        exit_loop_tools: Optional[List[ChatCompletionInputTool]] = None,
        exit_if_first_chunk_no_tool: bool = False,
    ) -> AsyncIterable[Union[ChatCompletionStreamOutput, ChatCompletionInputMessage]]:
        """Process a query using `self.model` and available tools, yielding chunks and tool outputs.

        Args:
            messages (`List[Dict]`):
                List of message objects representing the conversation history
            exit_loop_tools (`List[ChatCompletionInputTool]`, *optional*):
                List of tools that should exit the generator when called
            exit_if_first_chunk_no_tool (`bool`, *optional*):
                Exit if no tool is present in the first chunks. Default to False.

        Yields:
            [`ChatCompletionStreamOutput`] chunks or [`ChatCompletionInputMessage`] objects
        """
        # Prepare tools list based on options
        tools = self.available_tools
        if exit_loop_tools is not None:
            tools = [*exit_loop_tools, *self.available_tools]

        # Create the streaming request
        response = await self.client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        message = {"role": "unknown", "content": ""}
        final_tool_calls: Dict[int, ChatCompletionStreamOutputDeltaToolCall] = {}
        num_of_chunks = 0

        # Read from stream
        async for chunk in response:
            # Yield each chunk to caller
            yield chunk

            num_of_chunks += 1
            delta = chunk.choices[0].delta if chunk.choices and len(chunk.choices) > 0 else None
            if not delta:
                continue

            # Process message
            if delta.role:
                message["role"] = delta.role
            if delta.content:
                message["content"] += delta.content

            # Process tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # Aggregate chunks into tool calls
                    if tool_call.index not in final_tool_calls:
                        if tool_call.function.arguments is None:  # Corner case (depends on provider)
                            tool_call.function.arguments = ""
                        final_tool_calls[tool_call.index] = tool_call

                    if tool_call.function.arguments:
                        final_tool_calls[tool_call.index].function.arguments += tool_call.function.arguments

            # Optionally exit early if no tools in first chunks
            if exit_if_first_chunk_no_tool and num_of_chunks <= 2 and len(final_tool_calls) == 0:
                return

        if message["content"]:
            messages.append(message)

        # Process tool calls one by one
        for tool_call in final_tool_calls.values():
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments or "{}")

            tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": "", "name": function_name}

            # Check if this is an exit loop tool
            if exit_loop_tools and function_name in [t.function.name for t in exit_loop_tools]:
                tool_message_as_obj = ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
                messages.append(tool_message_as_obj)
                yield tool_message_as_obj
                return

            # Execute tool call with the appropriate session
            session = self.sessions.get(function_name)
            if session is not None:
                result = await session.call_tool(function_name, function_args)
                tool_message["content"] = format_result(result)
            else:
                error_msg = f"Error: No session found for tool: {function_name}"
                tool_message["content"] = error_msg

            # Yield tool message
            tool_message_as_obj = ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
            messages.append(tool_message_as_obj)
            yield tool_message_as_obj

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
