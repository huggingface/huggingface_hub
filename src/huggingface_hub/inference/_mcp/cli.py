import asyncio
import os
import signal
from functools import partial
from typing import Any, Dict, List, Optional

import typer
from rich import print

from .agent import Agent
from .utils import _load_agent_config


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
) -> None:
    """
    Tiny Agent loop.

    Args:
        agent_path (`str`, *optional*):
            Path to a local folder containing an `agent.json` and optionally a custom `PROMPT.md` file or a built-in agent stored in a Hugging Face dataset.

    """
    config, prompt = _load_agent_config(agent_path)

    servers: List[Dict[str, Any]] = config.get("servers", [])

    abort_event = asyncio.Event()
    first_sigint = True

    loop = asyncio.get_running_loop()
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler() -> None:
        nonlocal first_sigint
        if first_sigint:
            first_sigint = False
            abort_event.set()
            print("\n[red]Interrupted. Press Ctrl+C again to quit.[/red]", flush=True)
            return

        print("\n[red]Exiting...[/red]", flush=True)

        os._exit(130)

    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)

        async with Agent(
            provider=config["provider"],
            model=config["model"],
            servers=servers,
            prompt=prompt,
        ) as agent:
            await agent.load_tools()
            print(f"[bold blue]Agent loaded with {len(agent.available_tools)} tools:[/bold blue]")
            for t in agent.available_tools:
                print(f"[blue] • {t.function.name}[/blue]")

            while True:
                abort_event.clear()

                try:
                    user_input = await _ainput()
                    first_sigint = True
                except EOFError:
                    print("\n[red]EOF received, exiting.[/red]", flush=True)
                    break
                except KeyboardInterrupt:
                    if not first_sigint and abort_event.is_set():
                        continue
                    else:
                        print("\n[red]Keyboard interrupt during input processing.[/red]", flush=True)
                        break

                try:
                    async for chunk in agent.run(user_input, abort_event=abort_event):
                        if abort_event.is_set() and not first_sigint:
                            break

                        if hasattr(chunk, "choices"):
                            delta = chunk.choices[0].delta
                            if delta.content:
                                print(delta.content, end="", flush=True)
                            if delta.tool_calls:
                                for call in delta.tool_calls:
                                    if call.id:
                                        print(f"<Tool {call.id}>", end="")
                                    if call.function.name:
                                        print(f"{call.function.name}", end=" ")
                                    if call.function.arguments:
                                        print(f"{call.function.arguments}", end="")
                        else:
                            print(
                                f"\n\n[green]Tool[{chunk.name}] {chunk.tool_call_id}\n{chunk.content}[/green]\n",
                                flush=True,
                            )

                    print()

                except Exception as e:
                    print(f"\n[bold red]Error during agent run: {e}[/bold red]", flush=True)
                    first_sigint = True  # Allow graceful interrupt for the next command

    finally:
        if loop and not loop.is_closed():
            loop.remove_signal_handler(signal.SIGINT)
        elif original_sigint_handler:
            signal.signal(signal.SIGINT, original_sigint_handler)


@run_cli.callback()
def run(
    path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to a local folder containing an agent.json file or a built-in agent "
            "stored in the 'tiny-agents/tiny-agents' Hugging Face dataset "
            "(https://huggingface.co/datasets/tiny-agents/tiny-agents)"
        ),
        show_default=False,
    ),
):
    try:
        asyncio.run(run_agent(path))
    except KeyboardInterrupt:
        print("\n[red]Application terminated by KeyboardInterrupt.[/red]", flush=True)
        raise typer.Exit(code=130)
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]", flush=True)
        raise e


if __name__ == "__main__":
    app()
