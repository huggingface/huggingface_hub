import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from huggingface_hub import AsyncInferenceClient, ChatCompletionInputTool, ChatCompletionOutput
from huggingface_hub.inference._providers import PROVIDER_T


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
        # Initialize MCP session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools: List[ChatCompletionInputTool] = []

    async def add_mcp_server(self, command: str, args: List[str]):
        """Connect to an MCP server

        Args:
            todo
        """
        server_params = StdioServerParameters(command=command, args=args, env={"HF_TOKEN": os.environ["HF_TOKEN"]})

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
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
        """Process a query using Claude and available tools"""
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

                # Execute tool call
                result = await self.session.call_tool(function_name, function_args)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result.content[0].text,
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
