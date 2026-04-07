# Hugging Face Hub Web UI

A web-based interface for executing Hugging Face Hub CLI commands from your browser.

## Overview

The Web UI provides a user-friendly form-based interface to run all `hf` CLI commands without needing to use the terminal directly. Features include:

- **Form-based command interface**: Dropdowns and input fields for every CLI command
- **Real-time output streaming**: Watch command output appear as it executes
- **WebSocket support**: Bidirectional communication for command execution and output streaming
- **localhost-only binding**: Secure by default (localhost only)
- **Environment token support**: Automatically uses `HF_TOKEN` from environment

## Architecture

### Backend Structure

```
src/huggingface_hub/_web_ui/
├── __init__.py              # Package initialization
├── server.py                # FastAPI application with WebSocket support
├── command_executor.py      # Command execution with streaming
└── command_registry.py      # Registry of available CLI commands
```

### Frontend

The frontend is an embedded single-page application (SPA) served directly from FastAPI with:
- Command category browser
- Dynamic form generation based on command parameters
- Real-time output display in a terminal-like interface
- WebSocket streaming for live command output

## Usage

### Starting the Web UI

```bash
# Start with default settings (localhost:7860, auto-open browser)
hf web-ui run

# Custom host/port
hf web-ui run --host 0.0.0.0 --port 8000

# Don't auto-open browser
hf web-ui run --no-open-browser
```

Then open your browser to the displayed URL (default: `http://localhost:7860`)

### Using the Interface

1. **Select a category** from the dropdown menu
2. **Choose a command** from the category cards
3. **Fill in required arguments** in the form fields
4. **Select optional flags** with checkboxes and provide values
5. **Click "Execute Command"** to run
6. **Watch the output** stream in real-time

## Components

### Command Executor (`command_executor.py`)

Handles command execution with async streaming:
- Spawns subprocess for `hf` CLI commands
- Captures stdout/stderr in real-time
- Yields output line-by-line for streaming to WebSocket clients
- Handles process exit codes and errors

### Command Registry (`command_registry.py`)

Maintains metadata for all CLI commands:
- Organized by category (Authentication, Models, Datasets, etc.)
- Command descriptions
- Required arguments and optional flags
- Extensible for adding new commands

### FastAPI Server (`server.py`)

Core application serving:
- REST endpoints for command metadata
- WebSocket endpoint for command execution
- Embedded HTML/CSS/JavaScript frontend (SPA)
- CORS support for browser compatibility
- Connection management for multiple clients

## API Endpoints

### REST Endpoints

- `GET /api/health` - Health check
- `GET /api/commands` - Get all available commands by category

### WebSocket

- `WS /ws/execute` - Execute command with streaming output
  - Send: `{"command": ["repo", "ls", "username/repo"]}`
  - Receive: `{"output": "..."}` (repeatedly) or `{"status": "completed"}` or `{"error": "..."}`

## Example Workflow: Listing Models

1. Open Web UI (`http://localhost:7860`)
2. Select "Models" category
3. Select "models list" command
4. (Optional) Check "Search" flag and enter search term
5. Click "Execute Command"
6. View streaming output of available models

## Configuration

### Environment Variables

- `HF_TOKEN` - Hugging Face API token (auto-detected from environment)
- `HF_DEBUG` - Enable debug logging (inherited from main hf CLI)

### Command-line Options

```bash
hf web-ui run --help
```

Options:
- `--host` (default: 127.0.0.1) - Bind address
- `--port` (default: 7860) - Bind port  
- `--open-browser/--no-open-browser` (default: True) - Auto-open browser

## Command Registry Coverage

The registry is built dynamically at runtime by introspecting the live Typer CLI app.

- Automatically discovers top-level commands and command groups.
- Recursively traverses nested Typer groups.
- Extracts positional arguments and option flags from command signatures.
- Skips hidden commands/groups.

This means new CLI commands are automatically available in the web UI without manual updates to a static list.

## Technical Details

### Request Flow

1. User selects command and fills form in browser
2. JavaScript sends command array via WebSocket: `["repo", "ls", "model_id"]`
3. Server receives command and executes: `hf repo ls model_id`
4. Subprocess output is streamed back to client in real-time
5. Browser displays output as it arrives

### Dependencies

- **FastAPI** - Web framework (already in project)
- **uvicorn** - ASGI server (in oauth extras)
- **typer** - CLI framework (already in project)

### Security

- **Localhost-only by default**: Only accepts connections from localhost for security
- **Token handling**: Uses existing environment HF_TOKEN
- **Command validation**: Only allows registered CLI commands

## Future Enhancements

Potential features for future versions:

- Command history and favorites
- Export results to file
- Command syntax documentation in sidebar
- Batch command execution
- Multiple command profiles/workspaces
- Response filtering and formatting
- API key management UI
- Command scheduling

## Troubleshooting

### "Address already in use"
The default port 7860 is in use. Try a different port:
```bash
hf web-ui run --port 8000
```

### "Connection refused" in browser
Ensure the server is running and you're using the correct URL shown in terminal.

### Commands not executing
- Check that HF_TOKEN is set (if command requires authentication)
- Verify the command syntax in the command registry
- Check server logs for errors

## Development

To modify the command registry, edit `src/huggingface_hub/_web_ui/command_registry.py`:

```python
COMMANDS = {
    "Category Name": [
        {
            "name": "subcommand name",
            "description": "What this command does",
            "args": ["required_arg1", "required_arg2"],
            "flags": ["--optional-flag1", "--optional-flag2"]
        }
    ]
}
```

## Related Documentation

- [Hugging Face Hub CLI Reference](../cli/)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
