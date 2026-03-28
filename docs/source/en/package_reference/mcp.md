# MCP Client

The `huggingface_hub` library now includes an [`MCPClient`], designed to empower Large Language Models (LLMs) with the ability to interact with external Tools via the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). This client extends an [`AsyncInferenceClient`] to seamlessly integrate Tool usage.

The [`MCPClient`] connects to MCP servers (local `stdio` scripts or remote `http`/`sse` services) that expose tools. It feeds these tools to an LLM (via [`AsyncInferenceClient`]). If the LLM decides to use a tool, [`MCPClient`] manages the execution request to the MCP server and relays the Tool's output back to the LLM, often streaming results in real-time.

We also provide a higher-level [`Agent`] class. This 'Tiny Agent' simplifies creating conversational Agents by managing the chat loop and state, acting as a wrapper around [`MCPClient`].



## MCP Client

[[autodoc]] MCPClient

## Agent

[[autodoc]] Agent