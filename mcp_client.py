import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Dict, List, Optional, TypeAlias

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from huggingface_hub import AsyncInferenceClient, ChatCompletionInputTool, ChatCompletionOutput
from huggingface_hub.inference._providers import PROVIDER_T


# Type alias for tool names
ToolName: TypeAlias = str


class MCPClient(AsyncInferenceClient):
    def __init__(
        self,
        *,
        provider: PROVIDER_T,
        model: str,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            provider=provider,
            api_key=api_key,
        )
        self.model = model
        # Initialize MCP sessions as a dictionary of ClientSession objects
        self.sessions: Dict[ToolName, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.available_tools: List[ChatCompletionInputTool] = []

    async def add_mcp_server(self, command: str, args: List[str]):
        """Connect to an MCP server

        Args:
            todo
        """
        server_params = StdioServerParameters(command=command, args=args, env={"HF_TOKEN": os.environ["HF_TOKEN"]})

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Map tool names to their server for later lookup
        for tool in tools:
            self.sessions[tool.name] = session

        self.available_tools += [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    async def process_query(self, query: str) -> ChatCompletionOutput:
        """Process a query using `self.model` and available tools"""
        messages = [{"role": "user", "content": query}]

        response = await self.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.available_tools,
            tool_choice="auto",
        )

        # Process response and handle tool calls
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Get the appropriate session for this tool
                session = self.sessions.get(function_name)
                if session:
                    # Execute tool call with the appropriate session
                    result = await session.call_tool(function_name, function_args)
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": result.content[0].text,
                        }
                    )
                else:
                    error_msg = f"No session found for tool: {function_name}"
                    print(f"Error: {error_msg}")
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {error_msg}",
                        }
                    )

        function_enriched_response = await self.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return function_enriched_response

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient(
        provider="together",
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=os.environ["HF_TOKEN"],
    )
    try:
        await client.add_mcp_server(
            "node", ["--disable-warning=ExperimentalWarning", f"{os.path.expanduser('~')}/Desktop/hf-mcp/index.ts"]
        )
        response = await client.process_query(
            """
            find an app that generates 3D models from text,
            and also get the best paper about transformers
            """
        )
        print("\n" + response.choices[0].message.content)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
