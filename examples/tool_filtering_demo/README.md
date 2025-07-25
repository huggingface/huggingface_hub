# Tool Filtering Demo

This example demonstrates the new tool filtering feature for tiny agents.

## Configuration

The `agent.json` shows how to filter tools from MCP servers:

```json
{
  "servers": [
    {
      "type": "stdio",
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "tools": {
        "include": ["browser_click", "browser_close"]
      }
    }
  ]
}
```

## Tool Filtering Options

### Include only specific tools
```json
"tools": {
  "include": ["tool1", "tool2", "tool3"]
}
```

### Exclude specific tools  
```json
"tools": {
  "exclude": ["unwanted_tool1", "unwanted_tool2"]
}
```

### Combine both (exclude takes precedence)
```json
"tools": {
  "include": ["tool1", "tool2", "tool3"],
  "exclude": ["tool2"]
}
```
Result: Only `tool1` and `tool3` will be available.

## Running the Example

```bash
tiny-agents run examples/tool_filtering_demo
```

This agent will have access to only the `browser_click` and `browser_close` tools from Playwright, instead of all 30+ tools that Playwright provides by default.