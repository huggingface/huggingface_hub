import asyncio
import signal
from functools import partial
from typing import Any, Dict, List, Optional

import typer
from rich import print

from huggingface_hub.utils import get_token

from .agent import Agent
from .constants import DEFAULT_REPO_ID
from .utils import _load_agent_config, _url_to_server_config


app = typer.Typer(
    rich_markup_mode="rich",
    help="A squad of lightweight composable AI applications built on Hugging Face's Inference Client and MCP stack.",
)

run_cli = typer.Typer(
    name="run",
    help="Run the Agent in the CLI",
    invoke_without_command=True,
)
app.add_typer(run_cli, name="run")


async def _ainput(prompt: str = "» ") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(typer.prompt, prompt, prompt_suffix=" "))


async def run_agent(
    agent_path: Optional[str],
    *,
    extra_urls: Optional[List[str]],
    repo_id: Optional[str],
) -> None:
    """
    Tiny Agent loop.

    Args:
        agent_path (`str`, *optional*):
            Path to a local folder containing an `agent.json` and optionally a custom `PROMPT.md` file or a built-in agent stored in a Hugging Face dataset.
        extra_urls (`List[str]`, *optional*):
            List of URLs to override MCP servers.
        repo_id (`str`, *optional*):
            Hugging Face dataset repo containing agents. Defaults to `"huggingface/tiny-agents"`.

    """
    if repo_id is None:
        repo_id = DEFAULT_REPO_ID

    config, prompt = _load_agent_config(agent_path, repo_id)
    token = get_token()

    servers: List[Dict[str, Any]] = config.get("servers", [])
    if extra_urls:
        servers = [_url_to_server_config(u, token) for u in extra_urls]

    abort_event = asyncio.Event()
    first_sigint = True

    def _sigint_handler() -> None:
        nonlocal first_sigint
        if first_sigint:
            first_sigint = False
            abort_event.set()
            print("[grey]Interrupted – press Ctrl+C again to quit[/grey]")
        else:
            raise KeyboardInterrupt

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _sigint_handler)

    async with Agent(
        provider=config["provider"],
        model=config["model"],
        api_key=token,
        servers=servers,
        prompt=prompt,
    ) as agent:
        await agent.load_tools()
        print(f"[bold blue]Agent loaded with {len(agent.available_tools)} tools:[/bold blue]")
        for t in agent.available_tools:
            print(f"[blue] • {t.function.name}[/blue]")

        while True:
            try:
                user_input = await _ainput()
            except EOFError:
                break
            except KeyboardInterrupt:
                raise

            abort_event.clear()
            first_sigint = True

            async for chunk in agent.run(user_input, abort_event=abort_event):
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(delta.content, end="", flush=True)
                    if delta.tool_calls:
                        for call in delta.tool_calls:
                            if call.id:
                                print(f"[grey]<Tool {call.id}>[/grey]", end="")
                            if call.function.name:
                                print(f"[grey]{call.function.name}[/grey]", end=" ")
                            if call.function.arguments:
                                print(f"[grey]{call.function.arguments}[/grey]", end="")
                else:
                    print(f"\n\n[green]Tool[{chunk.name}] {chunk.tool_call_id}\n{chunk.content}[/green]\n")
            print()


@run_cli.callback()
def run(
    path: Optional[str] = typer.Argument(
        None,
        help="Path to a local folder containing an agent.json file or a built-in agent stored in a Hugging Face dataset (default: https://huggingface.co/datasets/huggingface/tiny-agents)",
    ),
    repo_id: Optional[str] = typer.Option(
        DEFAULT_REPO_ID,
        "--repo-id",
        "-r",
        help="Hugging Face dataset repo containing agents",
        show_default=True,
    ),
    url: List[str] = typer.Option(
        None,
        "--url",
        "-u",
        help="Override MCP servers by URL",
        show_default=False,
    ),
):
    try:
        asyncio.run(run_agent(path, extra_urls=url, repo_id=repo_id))
    except KeyboardInterrupt:
        raise typer.Exit(code=130)


if __name__ == "__main__":
    app()
