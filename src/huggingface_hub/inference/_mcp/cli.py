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
import signal
import sys
from functools import partial
from typing import Any, Dict, List, Optional

from colorama import Fore, Style
from colorama import init as colorama_init

from huggingface_hub.utils import get_token

from .agent import Agent
from .utils import _load_config, _url_to_server_config


colorama_init()
BLUE, GREEN, GRAY, RESET = Fore.BLUE, Fore.GREEN, Fore.LIGHTBLACK_EX, Style.RESET_ALL


async def ainput(prompt: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(input, prompt))


async def run_agent(source: Optional[str], extra_urls: Optional[List[str]]) -> None:
    config, prompt = _load_config(source)
    hf_token = get_token()

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
