Tiny agents (https://huggingface.co/blog/python-tiny-agents) is a minimalistic framework for running AI agents. When running a tiny  agent with `huggingface_hub` using the  `tiny-agents run agent` command, the command will look for an `agent.json` file which  defines the configuration of the agent. Each agent is defined by an LLM (powered by  Hugging Face Inference Providers which is similar to the OpenAI API) as well as a set of MCP servers, whose tools will be provided to the LLM. Currently, one can just add certain MCP servers to the config, such as the one below:

```json
{
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "provider": "nebius",
  "inputs": [
    {
      "type": "promptString",
      "id": "github-personal-access-token",
      "description": "Github Personal Access Token (read-only)",
      "password": true
    }
  ],
  "servers": [
    {
     "type": "stdio",
     "command": "docker",
     "args": [
       "run",
       "-i",
       "--rm",
       "-e",
       "GITHUB_PERSONAL_ACCESS_TOKEN",
       "-e",
       "GITHUB_TOOLSETS=repos,issues,pull_requests",
       "ghcr.io/github/github-mcp-server"
     ],
     "env": {
       "GITHUB_PERSONAL_ACCESS_TOKEN": "${input:github-personal-access-token}"
     }
    },
    {
      "type": "stdio",
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    }
  ]
}
```

However it would be nice to have a feature that allows users to define which tools to enable/disable in the config JSON file. For example, for the Playwright MCP server (which by default has more than 30 tools), I actually only need the `browser_click` and `browser_close` tools. Enabling only a handful of tools makes AI agents much more reliable.

Would you be able to implement this feature?