# MCP Client & Tiny Agent (experimental)

The `huggingface_hub` library now includes an experimental [`MCPClient`], designed to empower Large Language Models (LLMs) with the ability to interact with external Tools via the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). This client extends an [`AsyncInferenceClient`] to seamlessly integrate Tool usage.

The [`MCPClient`] connects to MCP servers (local `stdio` scripts or remote `http`/`sse` services) that expose tools. It feds these tools to an LLM (via [`AsyncInferenceClient`]). If the LLM decides to use a tool, [`MCPClient`] manages the execution request to the MCP server and relays the Tool's output back to the LLM, often streaming results in real-time.

In the following example, we use [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model via [Nebius](https://nebius.com/) inference provider. We then add an MCP server, in this case, an SSE server which made the Flux image generation tool available to the LLM.


We also offer a higher-level [`Agent`] class. This 'Tiny Agent' simplifies creating conversational Agents by managing the chat loop and state, essentially acting as a wrapper around [`MCPClient`]. It's designed to be a simple while loop built right on top of an [`MCPClient`].


## MCP Client

[[autodoc]] MCPClient

## Agent

[[autodoc]] Agent