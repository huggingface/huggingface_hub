# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI server for the web UI."""

# ruff: noqa: W293

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from huggingface_hub._web_ui.command_executor import EXIT_CODE_PREFIX, CommandExecutor
from huggingface_hub._web_ui.command_registry import CommandRegistry


# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


def _is_allowed_websocket_origin(websocket: WebSocket) -> bool:
    """Return True when the WebSocket Origin matches the current request host."""
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")

    if not origin or not host:
        return False

    parsed_origin = urlparse(origin)
    parsed_host = urlparse(f"//{host}")

    return (
        parsed_origin.scheme in {"http", "https"}
        and parsed_origin.hostname == parsed_host.hostname
        and parsed_origin.port == parsed_host.port
    )


def _resolve_frontend_static_path(full_path: str, frontend_root: Path | None = None) -> Path | None:
    """Return a frontend file path only if it stays within the frontend root."""
    root = (frontend_root or Path(__file__).parent / "frontend").resolve()
    candidate = (root / full_path).resolve()

    if not candidate.is_file() or not candidate.is_relative_to(root):
        return None

    return candidate


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        print("\nHugging Face Web UI is running.")
        print("Open your browser and go to: http://localhost:7860")
        yield
        # Shutdown
        print("\nWeb UI shutting down...")

    app = FastAPI(
        title="Hugging Face Hub Web UI",
        description="Web interface for Hugging Face Hub CLI commands",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routes
    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/api/commands")
    async def get_commands():
        """Get all available commands organized by category."""
        return {"commands": CommandRegistry.get_all_commands()}

    @app.websocket("/ws/execute")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for command execution with streaming output."""
        if not _is_allowed_websocket_origin(websocket):
            await websocket.close(code=1008)
            return

        await manager.connect(websocket)
        try:
            while True:
                # Receive command from client
                data = await websocket.receive_json()
                command_args = data.get("command", [])

                if not command_args:
                    await websocket.send_json({"error": "No command provided"})
                    continue

                # Execute using the current Python environment to avoid PATH issues for `hf`.
                full_command = [sys.executable, "-m", "huggingface_hub.cli.hf"] + command_args

                # Execute command and stream output
                try:
                    exit_code = 0
                    async for output in CommandExecutor.execute_command(full_command):
                        if output.startswith(EXIT_CODE_PREFIX):
                            exit_code = int(output.removeprefix(EXIT_CODE_PREFIX))
                            continue

                        if output:
                            await websocket.send_json({"output": output})

                    if exit_code == 0:
                        await websocket.send_json({"status": "completed"})
                    else:
                        await websocket.send_json({"status": "failed", "exit_code": exit_code})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            manager.disconnect(websocket)

    # Serve static frontend assets
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        """Serve the main HTML file."""
        return get_fallback_html()

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_static(full_path: str):
        """Serve static files or fallback to index.html for SPA routing."""
        static_path = _resolve_frontend_static_path(full_path)
        if static_path is not None:
            return FileResponse(static_path)

        # Fallback to index.html for SPA
        if full_path and "." not in full_path:
            return get_fallback_html()

        return HTMLResponse(status_code=404)

    return app


def get_fallback_html() -> str:
    """Get the fallback HTML for the SPA."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Hugging Face Hub Web UI</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .container {
                max-width: 1200px;
                width: 100%;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 28px;
                margin-bottom: 10px;
            }
            
            .content {
                padding: 40px;
            }
            
            .command-section {
                margin-bottom: 30px;
            }
            
            .category {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .category:hover {
                background: #e9ecef;
                transform: translateX(5px);
            }
            
            .category h3 {
                color: #667eea;
                font-size: 16px;
                margin-bottom: 10px;
            }
            
            .commands-list {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .command-card {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 15px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .command-card:hover {
                border-color: #667eea;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
                transform: translateY(-2px);
            }
            
            .command-card h4 {
                color: #667eea;
                font-size: 14px;
                margin-bottom: 5px;
            }
            
            .command-card p {
                color: #6c757d;
                font-size: 13px;
                line-height: 1.4;
            }
            
            .input-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 20px;
            }
            
            .input-group {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            input, select {
                flex: 1;
                padding: 10px;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                font-size: 14px;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .output-section {
                background: #1e1e1e;
                color: #00ff00;
                padding: 20px;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                max-height: 400px;
                overflow-y: auto;
                line-height: 1.5;
            }
            
            .output-line {
                margin-bottom: 5px;
            }
            
            .loading {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #00ff00;
                border-radius: 50%;
                animation: blink 1s infinite;
            }
            
            @keyframes blink {
                0%, 49%, 100% { opacity: 1; }
                50%, 99% { opacity: 0; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤗 Hugging Face Hub Web UI</h1>
                <p>Execute CLI commands directly from your browser</p>
            </div>
            
            <div class="content">
                <div class="input-section">
                    <h3>Select Command</h3>
                    <div class="input-group">
                        <select id="categorySelect">
                            <option value="">-- Select a category --</option>
                        </select>
                    </div>
                    <div class="commands-list" id="commandsList"></div>
                </div>
                
                <div class="input-section">
                    <h3>Command Arguments</h3>
                    <div id="argsSection"></div>
                    <button onclick="executeCommand()">Execute Command</button>
                </div>
                
                <div>
                    <h3>Output</h3>
                    <div class="output-section" id="output">
                        <div class="output-line">Ready to execute commands...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedCommand = null;
            let ws = null;
            
            async function init() {
                try {
                    const response = await fetch('/api/commands');
                    const data = await response.json();
                    const commands = data.commands;
                    
                    const categorySelect = document.getElementById('categorySelect');
                    Object.keys(commands).forEach(category => {
                        const option = document.createElement('option');
                        option.value = category;
                        option.textContent = category;
                        categorySelect.appendChild(option);
                    });
                    
                    categorySelect.addEventListener('change', (e) => showCommands(e.target.value, commands));
                } catch (error) {
                    console.error('Error loading commands:', error);
                    addOutput('Error loading commands: ' + error.message);
                }
            }
            
            function showCommands(category, commands) {
                const commandsList = document.getElementById('commandsList');
                commandsList.innerHTML = '';
                
                if (!category) return;
                
                commands[category].forEach(cmd => {
                    const card = document.createElement('div');
                    card.className = 'command-card';
                    card.innerHTML = `
                        <h4>${cmd.name}</h4>
                        <p>${cmd.description}</p>
                    `;
                    card.onclick = () => selectCommand(cmd);
                    commandsList.appendChild(card);
                });
            }
            
            function selectCommand(cmd) {
                selectedCommand = cmd;
                renderCommandForm(cmd);
            }
            
            function renderCommandForm(cmd) {
                const argsSection = document.getElementById('argsSection');
                argsSection.innerHTML = '';
                
                if (cmd.args && cmd.args.length > 0) {
                    cmd.args.forEach(arg => {
                        const group = document.createElement('div');
                        group.className = 'input-group';
                        group.innerHTML = `
                            <label style="flex-basis: 100%; margin-bottom: 5px; font-weight: bold; color: #333;">${arg}:</label>
                            <input type="text" id="arg_${arg}" placeholder="Enter ${arg}" style="flex-basis: 100%; margin-bottom: 10px;">
                        `;
                        argsSection.appendChild(group);
                    });
                }
                
                if (cmd.flags && cmd.flags.length > 0) {
                    const flagsDiv = document.createElement('div');
                    flagsDiv.innerHTML = '<p style="margin-bottom: 10px; color: #666;">Optional flags:</p>';
                    cmd.flags.forEach(flag => {
                        const group = document.createElement('div');
                        group.className = 'input-group';
                        group.innerHTML = `
                            <label style="display: flex; align-items: center; flex: 1;">
                                <input type="checkbox" id="flag_${flag}" style="width: auto; margin-right: 10px;">
                                <span>${flag}</span>
                            </label>
                            <input type="text" id="flag_val_${flag}" placeholder="value" style="flex: 1;">
                        `;
                        flagsDiv.appendChild(group);
                    });
                    argsSection.appendChild(flagsDiv);
                }
            }
            
            function executeCommand() {
                if (!selectedCommand) {
                    addOutput('Please select a command first');
                    return;
                }
                
                const command = selectedCommand.name.split(' ');
                
                // Add arguments
                if (selectedCommand.args) {
                    selectedCommand.args.forEach(arg => {
                        const value = document.getElementById(`arg_${arg}`)?.value;
                        if (value) command.push(value);
                    });
                }
                
                // Add flags
                if (selectedCommand.flags) {
                    selectedCommand.flags.forEach(flag => {
                        const checked = document.getElementById(`flag_${flag}`)?.checked;
                        if (checked) {
                            command.push(flag);
                            const value = document.getElementById(`flag_val_${flag}`)?.value;
                            if (value) command.push(value);
                        }
                    });
                }
                
                clearOutput();
                addOutput(`Executing: hf ${command.join(' ')}`);
                addOutput('---');
                
                executeViaWebSocket(command);
            }
            
            function executeViaWebSocket(command) {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws/execute`;
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    ws.send(JSON.stringify({ command }));
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.output) {
                        addOutput(data.output);
                    } else if (data.status === 'completed') {
                        addOutput('---');
                        addOutput('Command completed successfully');
                    } else if (data.status === 'failed') {
                        addOutput('---');
                        addOutput('Command failed (exit code ' + data.exit_code + ')');
                    } else if (data.error) {
                        addOutput('Error: ' + data.error);
                    }
                };
                
                ws.onerror = () => {
                    addOutput('WebSocket error: unable to connect to backend (' + wsUrl + ')');
                };
                
                ws.onclose = () => {
                    // Connection closed
                };
            }
            
            function addOutput(text) {
                const output = document.getElementById('output');
                const line = document.createElement('div');
                line.className = 'output-line';
                line.textContent = text;
                output.appendChild(line);
                output.scrollTop = output.scrollHeight;
            }
            
            function clearOutput() {
                document.getElementById('output').innerHTML = '';
            }
            
            init();
        </script>
    </body>
    </html>
    """
