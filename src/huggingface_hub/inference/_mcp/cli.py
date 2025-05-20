"""
Tiny Agents CLI

A squad of lightweight composable AI applications built on Hugging Face's Inference Client and MCP stack.
Usage:
    pip install huggingface-hub[mcp]
    tiny-agents run <path> [--url <remote-url>]...

Arguments:
    <path>       Path to an agent folder, a standalone agent.json file, or the name of a built-in agent.
    --url        (Optional) One or more remote MCP server URLs to override the servers defined in agent.json.

Examples:
    tiny-agents run my-agent
    tiny-agents run agent.json
"""

import argparse
import asyncio
import importlib.resources as importlib_resources
import json
import os
import signal
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from colorama import Fore, Style
from colorama import init as colorama_init

from .tiny_agent import Agent


FILENAME_CONFIG = "agent.json"
FILENAME_PROMPT = "PROMPT.md"

DEFAULT_AGENT = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "nebius",
    "servers": [
        {
            "type": "stdio",
            "config": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                ],
            },
        },
        {
            "type": "stdio",
            "config": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
            },
        },
    ],
}

colorama_init()
BLUE, GREEN, GRAY, RESET = Fore.BLUE, Fore.GREEN, Fore.LIGHTBLACK_EX, Style.RESET_ALL


def _load_builtin_agent_path(name: str) -> Optional[Path]:
    try:
        base = importlib_resources.files("huggingface_hub.inference._mcp.agents")
    except (ModuleNotFoundError, AttributeError):
        return None
    candidate = base / name
    return candidate if candidate.is_dir() else None


def _load_config(source: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load configuration and prompt."""
    if source is None:
        return (
            DEFAULT_AGENT,
            None,
        )

    path = Path(source).expanduser()

    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), None

    if path.is_dir():
        config_json = (path / FILENAME_CONFIG).read_text(encoding="utf-8")
        try:
            prompt_md = (path / FILENAME_PROMPT).read_text(encoding="utf-8")
        except FileNotFoundError:
            prompt_md = None
        return json.loads(config_json), prompt_md

    builtin_dir = _load_builtin_agent_path(source)
    if builtin_dir is not None:
        config_json = (builtin_dir / FILENAME_CONFIG).read_text(encoding="utf-8")
        try:
            prompt_md = (builtin_dir / FILENAME_PROMPT).read_text(encoding="utf-8")
        except FileNotFoundError:
            prompt_md = None
        return json.loads(config_json), prompt_md

    raise FileNotFoundError(source)


def _url_to_server_config(url: str, hf_token: Optional[str]) -> Dict:
    return {
        "command": None,
        "url": url,
        "env": {"AUTHORIZATION": f"Bearer {hf_token}"} if hf_token else None,
    }


async def ainput(prompt: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(input, prompt))


async def run_agent(source: Optional[str], extra_urls: Optional[List[str]]) -> None:
    config, prompt = _load_config(source)
    hf_token = os.getenv("HF_TOKEN")

    servers: List[Dict[str, Any]] = config.get("servers", [])
    if extra_urls:
        servers.clear()
        for url in extra_urls:
            try:
                servers.append(_url_to_server_config(url, hf_token))
            except Exception as e:
                sys.stderr.write(f'Error adding server "{url}": {e}\n')

    abort_event = asyncio.Event()
    _ctrl_c_count = 0

    def _sigint_handler() -> None:
        nonlocal _ctrl_c_count
        if _ctrl_c_count == 0 and not abort_event.is_set():
            _ctrl_c_count += 1
            abort_event.set()
            print(f"\n{GRAY}Press Ctrl+C again to exit{RESET}")
        else:
            print("\nExiting...")
            sys.exit(0)

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _sigint_handler)

    async with Agent(
        provider=config["provider"],
        model=config["model"],
        api_key=hf_token,
        servers=servers,
        prompt=prompt,
    ) as agent:
        await agent.load_tools()
        print(f"{BLUE}Agent loaded with {len(agent.available_tools)} tools:")
        print("\n".join(f"- {t.function.name}" for t in agent.available_tools))
        print(RESET, end="")

        while True:
            try:
                user_input = await ainput("» ")
            except (EOFError, KeyboardInterrupt):
                break

            abort_event.clear()
            _ctrl_c_count = 0

            async for chunk in agent.run(user_input, abort_event=abort_event):
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(delta.content, end="", flush=True)
                    if delta.tool_calls:
                        print(GRAY, end="")
                        for call in delta.tool_calls:
                            if call.id:
                                print(f"<Tool {call.id}>")
                            if call.function.name:
                                print(call.function.name, end=" ")
                            if call.function.arguments:
                                print(call.function.arguments, end="")
                        print(RESET, end="")
                else:
                    print(f"\n\n{GREEN}Tool[{chunk.name}] {chunk.tool_call_id}\n{chunk.content}{RESET}\n")

            print()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="tiny-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run agent from folder or builtin name")
    run_parser.add_argument("path", type=str, help="Path to folder, agent.json, or builtin name", nargs="?")
    run_parser.add_argument("--url", action="append", help="Override MCP servers by URL")

    args = parser.parse_args(argv)

    if args.command == "run":
        asyncio.run(run_agent(args.path, args.url))


if __name__ == "__main__":
    main()
