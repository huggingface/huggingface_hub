from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import timedelta
from textwrap import dedent
from typing import Any, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._generated.types.chat_completion import ChatCompletionOutputMessage
from huggingface_hub.inference._providers import PROVIDER_T
from huggingface_hub.utils import logging


logger = logging.get_logger(__name__)


class MCPServerBase:
    """Modified MCP implementation from any-agent https://github.com/mozilla-ai/any-agent/pull/168"""

    client: Any | None = None
    session: Any | None = None

    def __init__(self, mcp_tool: Any):
        self.mcp_tool = mcp_tool
        self.tools: list[Callable[..., Any]] = []
        self._exit_stack = AsyncExitStack()

    def _filter_tools(self, tools: Sequence[Any]) -> Sequence[Any]:
        # Only add the tools listed in mcp_tool['tools'] if specified
        requested_tools = list(self.mcp_tool.get('tools') or [])
        if not requested_tools:
            return tools
        tools = [tool for tool in tools if tool.name in requested_tools]
        if len(tools) != len(requested_tools):
            tool_names = [tool.name for tool in tools]
            raise ValueError(
                dedent(f"""Could not find all requested tools in the MCP server:
                            Requested: {requested_tools}
                            Set:   {tool_names}"""),
            )
        return tools

    async def _setup_tools(self) -> None:
        """Set up the MCP tools for TinyAgent."""
        if not self.client:
            msg = "MCP client is not set up. Please call `setup` from a concrete class."
            raise ValueError(msg)

        # Setup the client connection using exit stack to manage lifecycle
        stdio, write = await self._exit_stack.enter_async_context(self.client)

        # Create a client session
        client_session = ClientSession(
            stdio,
            write,
            timedelta(seconds=self.mcp_tool['client_session_timeout_seconds'])
            if self.mcp_tool.get('client_session_timeout_seconds')
            else None,
        )

        # Start the session
        self.session: ClientSession = await self._exit_stack.enter_async_context(client_session)
        if not self.session:
            msg = "Failed to create MCP session"
            raise ValueError(msg)
        await self.session.initialize()

        # Get the available tools from the MCP server using schema
        available_tools = await self.session.list_tools()

        # Filter tools if specific tools were requested
        filtered_tools = self._filter_tools(available_tools.tools)

        # Create callable tool functions
        tool_list = []
        for tool_info in filtered_tools:
            tool_list.append(self._create_tool_from_info(tool_info))

        # Store tools as a list
        self.tools = tool_list

    def _create_tool_from_info(self, tool) -> Callable[..., Any]:
        """Create a tool function from tool information."""
        tool_name = tool.name
        tool_description = tool.description
        input_schema = tool.inputSchema
        session = self.session
        if not self.session:
            msg = "Not connected to MCP server"
            raise ValueError(msg)

        async def tool_function(*args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
            """Tool function that calls the MCP server."""
            # Combine args and kwargs
            combined_args = {}
            if args and len(args) > 0:
                combined_args = args[0]
            combined_args.update(kwargs)

            if not session:
                msg = "Not connected to MCP server"
                raise ValueError(msg)
            # Call the tool on the MCP server
            try:
                return await session.call_tool(tool_name, combined_args)
            except Exception as e:
                return f"Error calling tool {tool_name}: {e!s}"

        # Set attributes for the tool function
        tool_function.__name__ = tool_name
        tool_function.__doc__ = tool_description
        # this isn't a defined attribute of a callable, but we pass it to tinyagent so that it can use it appropriately
        # when constructing the ToolExecutor.
        tool_function.__input_schema__ = input_schema  # type: ignore[attr-defined]

        return tool_function


class MCPServerStdio(MCPServerBase):
    """MCP adapter for Tiny framework using stdio communication."""

    mcp_tool: Any

    async def _setup_tools(self) -> None:
        server_params = StdioServerParameters(
            command=self.mcp_tool['command'],
            args=list(self.mcp_tool['args']),
            env={**os.environ},
        )

        self.client = stdio_client(server_params)

        await super()._setup_tools()


class MCPServerSse(MCPServerBase):
    """MCP adapter for Tiny framework using SSE communication."""

    mcp_tool: Any

    async def _setup_tools(self) -> None:
        self.client = sse_client(
            url=self.mcp_tool['url'],
            headers=dict(self.mcp_tool.get('headers') or {}),
        )

        await super()._setup_tools()


DEFAULT_SYSTEM_PROMPT = """
You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved, or if you need more info from the user to solve the problem.

If you are not sure about anything pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
""".strip()

# Max number of tool calling + chat completion steps in response to a single user query
DEFAULT_MAX_NUM_TURNS = 10


### Internal tools for tiny-agent ###"
def task_completion_tool() -> dict[str, Any]:
    """Tool to indicate task completion."""
    return {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this tool when the task given by the user is complete",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


class ToolExecutor:
    """Executor for tools that wraps tool functions to work with the MCP client."""

    def __init__(self, tool_function: Callable[..., Any]) -> None:
        """Initialize the tool executor.

        Args:
            tool_function: The tool function to execute

        """
        self.tool_function = tool_function

    async def call_tool(self, request: dict[str, Any]) -> dict[str, Any]:
        """Call the tool function.

        Args:
            request: The tool request with name and arguments

        Returns:
            Tool execution result

        """
        try:
            # Extract arguments
            arguments = request.get("arguments", {})

            # Call the tool function
            if asyncio.iscoroutinefunction(self.tool_function):
                result = await self.tool_function(**arguments)
            else:
                result = self.tool_function(**arguments)

            # Format the result
            if isinstance(result, str):
                return {"content": [{"text": result}]}
            if isinstance(result, dict):
                return {"content": [{"text": str(result)}]}
            return {"content": [{"text": str(result)}]}
        except Exception as e:
            return {"content": [{"text": f"Error executing tool: {e}"}]}


class TinyAgent:
    """A lightweight agent implementation using litellm.

    Modeled after JS implementation https://huggingface.co/blog/tiny-agents.
    """

    def __init__(
        self,
        config,
        provider: PROVIDER_T,
        model: str,
        api_key: str,
    ) -> None:
        """Initialize the TinyAgent.

        Args:
            config: Agent configuration
            managed_agents: Optional list of managed agent configurations
            tracing: Optional tracing configuration

        """
        self.client = AsyncInferenceClient(
            provider=provider,
            api_key=api_key,
        )
        self.model = model
        self.messages: list[dict[str, Any]] = []
        self.instructions = config["instructions"] or DEFAULT_SYSTEM_PROMPT
        self.clients: dict[str, ToolExecutor] = {}
        self.available_tools: list[dict[str, Any]] = []
        self.exit_loop_tools = [task_completion_tool()]
        self._mcp_servers = []
        self.mcp_tools = []

    async def add_stdio_mcp_server(self, command: str, args: list, tools: list[str], env_vars: dict) -> MCPServerBase:
        """Add an MCP server using SSE."""
        server =  MCPServerStdio(
            mcp_tool={
                "command": command,
                "args": args,
                "tools": tools,
                "env_vars": env_vars,
            },
        )
        self._mcp_servers.append(server)
        await server._setup_tools()
        return server

    async def add_sse_mcp_server(self, url: str, tools: list[str], env_vars: dict) -> MCPServerBase:
        """Add an MCP server using stdio."""
        server = MCPServerSse(
            mcp_tool={
                "url": url,
                "tools": tools,
                "env_vars": env_vars,
            },
        )
        self._mcp_servers.append(server)
        await server._setup_tools()
        return server

    async def load_agent(self, mcp_servers) -> None:
        """Load the agent and its tools."""
        # Load tools

        for server in mcp_servers:
            type_of_server = server.pop("type", None)
            if type_of_server == "stdio":
                mcp_server = await self.add_stdio_mcp_server(**server)
            elif type_of_server == "sse":
                mcp_server = await self.add_sse_mcp_server(**server)
            else:
                raise ValueError(f"Unsupported MCP server type: {server['type']}")
            for tool in mcp_server.tools:
                self.mcp_tools.append(tool)
        for tool in self.mcp_tools:
            tool_name = tool.__name__
            tool_desc = tool.__doc__ or f"Tool to {tool_name}"

            # check if the tool has __input__schema__ attribute which we set when wrapping MCP tools
            if not hasattr(tool, "__input_schema__"):
                # Generate one from the function signature
                import inspect

                sig = inspect.signature(tool)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    # Skip *args and **kwargs
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    # Add the parameter to properties
                    properties[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name}",
                    }

                    # If parameter has no default, it's required
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

                input_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            else:
                # Use the provided schema
                input_schema = tool.__input_schema__

            # Add the tool to available tools
            self.available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_desc,
                        "parameters": input_schema,
                    },
                }
            )

            # Register tool with the client
            self.clients[tool_name] = ToolExecutor(tool)

    async def run_async(self, prompt: str, **kwargs: Any):
        """Run the agent asynchronously.

        Args:
            prompt: User input prompt
            **kwargs: Additional parameters

        Returns:
            The final agent response

        """
        max_turns = kwargs.get("max_turns", DEFAULT_MAX_NUM_TURNS)
        self.messages = [
            {
                "role": "system",
                "content": self.instructions or DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        num_of_turns = 0
        next_turn_should_call_tools = True
        final_response = ""
        assistant_messages = []

        while True:
            try:
                logger.debug("Starting turn %s", num_of_turns + 1)

                last_response = await self._process_single_turn_with_tools(
                    {
                        "exit_loop_tools": self.exit_loop_tools,
                        "exit_if_first_chunk_no_tool": num_of_turns > 0 and next_turn_should_call_tools,
                    },
                )

                if last_response:
                    logger.debug(last_response)
                    logger.debug(
                        "Assistant response this turn: %s...",
                        last_response[:50],
                    )
                    assistant_messages.append(last_response)
                    final_response = last_response

            except Exception as err:
                logger.error("Error during turn %s: %s", num_of_turns + 1, err)
                if isinstance(err, Exception) and str(err) == "AbortError":
                    return {
                        "final_output": final_response or "Task aborted",
                        "raw_responses": self.messages,
                    }
                raise

            num_of_turns += 1
            current_last = self.messages[-1]
            logger.debug("Current role: %s", current_last.get("role"))

            # After a turn, check if we have any content in the last assistant message
            if current_last.get("role") == "assistant" and current_last.get("content"):
                final_response = current_last.get("content", "No content found")
                logger.debug(
                    "Updated final response from assistant message: %s...",
                    final_response[:50],
                )

            # Check exit conditions
            if (
                current_last.get("role") == "tool"
                and current_last.get("name")
                and current_last.get("name") in [t["function"]["name"] for t in self.exit_loop_tools]
            ):
                logger.debug("Exiting because tool %s is an exit tool", current_last.get("name"))
                # If task is complete, return the last assistant message before this
                for msg in reversed(self.messages[:-1]):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        return {
                            "final_output": msg.get("content"),
                            "raw_responses": self.messages,
                        }
                return {
                    "final_output": final_response or "Task completed",
                    "raw_responses": self.messages,
                }

            if current_last.get("role") != "tool" and num_of_turns > max_turns:
                logger.debug("Exiting because max turns (%s) reached", max_turns)
                return {
                    "final_output": final_response or "Max turns reached",
                    "raw_responses": self.messages,
                }

            if current_last.get("role") != "tool" and next_turn_should_call_tools:
                logger.debug("Exiting because no tools were called when expected")
                return {
                    "final_output": final_response or "No tools called",
                    "raw_responses": self.messages,
                }

            if current_last.get("role") == "tool":
                next_turn_should_call_tools = False
                logger.debug("Tool was called, next turn should not call tools")
            else:
                next_turn_should_call_tools = True
                logger.debug("No tool was called, next turn should call tools")

    async def _process_single_turn_with_tools(self, options: dict[str, Any]) -> str:
        """Process a single turn of conversation with potential tool calls.

        Args:
            options: Options including exit_loop_tools, exit_if_first_chunk_no_tool

        Returns:
            The response message or combined tool results

        """
        logger.debug("Start of single turn")

        tools = options["exit_loop_tools"] + self.available_tools

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto",
        )
        message: ChatCompletionOutputMessage = response.choices[0].message
        logger.debug("Response message: %s", dict(message))
        self.messages.append(dict(message))

        # Process tool calls if any
        combined_results = []
        exit_tool_called = False

        if message.tool_calls:
            logger.debug(f"Processing {len(message.tool_calls)} tool calls")

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                logger.debug("Processing tool call for: %s", tool_name)
                tool_args = {}

                if tool_call.function.arguments:
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.debug("Tool arguments: %s", tool_args)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "",
                    "name": tool_name,
                }

                # Check if tool is in exit loop tools
                exit_tools = options["exit_loop_tools"]
                if exit_tools and tool_name in [t["function"]["name"] for t in exit_tools]:
                    logger.debug("Exit tool called: %s", tool_name)
                    exit_tool_called = True
                    self.messages.append(tool_message)
                    combined_results.append(str(tool_message["content"]))
                    continue

                # Check if the tool exists
                if tool_name not in self.clients:
                    logger.error("Tool %s not found in registered tools", tool_name)
                    tool_message["content"] = f"Error: No tool found with name: {tool_name}"
                else:
                    client = self.clients[tool_name]
                    try:
                        logger.debug("Calling tool: %s", tool_name)
                        result = await client.call_tool({"name": tool_name, "arguments": tool_args})

                        if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
                            tool_message["content"] = result["content"][0]["text"]
                        else:
                            tool_message["content"] = str(result)

                        logger.debug(
                            "Tool result: %s...",
                            tool_message["content"][:50] if tool_message["content"] else "Empty",
                        )
                    except Exception as e:
                        logger.error("Error calling tool %s: %s", tool_name, e)
                        tool_message["content"] = f"Error calling tool {tool_name}: {e}"

                self.messages.append(tool_message)
                combined_results.append(str(tool_message["content"]))

            # If an exit tool was called, return early with the combined results
            if exit_tool_called:
                return "\n".join(combined_results)

            return "\n".join(combined_results)

        return str(message.content)
    async def cleanup(self):
        """Close all resources."""
        for server in self._mcp_servers:
            await server._exit_stack.aclose()

async def main():
    agent = TinyAgent(
        config={
            "instructions": DEFAULT_SYSTEM_PROMPT,
        },
        provider="together",
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=os.environ["HF_TOKEN"],
    )
    try:
        await agent.load_agent(
            [
                {
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["mcp-server-time", "--local-timezone=America/New_York"],
                    "tools": [
                        "get_current_time",
                    ],
                    "env_vars": {},
                },
            ]
        )
        response = await agent.run_async("What time is it? Please use the get_current_time tool to find out.", max_turns=5)
        print("Final response:", response["final_output"])
        print("Raw responses:", response["raw_responses"])
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
