from __future__ import annotations

import asyncio
import os
import platform
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Sequence, Union

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionInputTool, ChatCompletionStreamOutput, MCPClient


DEFAULT_SYSTEM_PROMPT = """
You are an agent - please keep going until the user’s query is completely
resolved, before ending your turn and yielding back to the user. Only terminate
your turn when you are sure that the problem is solved, or if you need more
info from the user to solve the problem.
If you are not sure about anything pertaining to the user’s request, use your
tools to read files and gather the relevant information: do NOT guess or make
up an answer.
You MUST plan extensively before each function call, and reflect extensively
on the outcomes of the previous function calls. DO NOT do this entire process
by making function calls only, as this can impair your ability to solve the
problem and think insightfully.
""".strip()

MAX_NUM_TURNS = 10

TASK_COMPLETE_TOOL: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj(
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this tool when the task given by the user is complete",
            "parameters": {"type": "object", "properties": {}},
        },
    }
)

ASK_QUESTION_TOOL: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj(
    {
        "type": "function",
        "function": {
            "name": "ask_question",
            "description": "Ask the user for more info required to solve or clarify their problem.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
)

EXIT_LOOP_TOOLS: List[ChatCompletionInputTool] = [TASK_COMPLETE_TOOL, ASK_QUESTION_TOOL]


MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
PROVIDER = os.getenv("PROVIDER", "nebius")

HOME = Path.home()

DESKTOP = HOME / "Desktop" if platform.system() == "Darwin" else HOME


class Agent(MCPClient):
    """
    Python implementation of the huggingface/mcp-client JS tiny agent
    """

    def __init__(
        self,
        *,
        model: str,
        servers: Sequence[Dict],
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(model=model, provider=provider, api_key=api_key)
        self._servers_cfg = list(servers)
        self.messages: List[Union[Dict, ChatCompletionInputMessage]] = [
            {"role": "system", "content": prompt or DEFAULT_SYSTEM_PROMPT}
        ]

    async def load_tools(self) -> None:
        for cfg in self._servers_cfg:
            await self.add_mcp_server(cfg)

    async def run(
        self,
        user_input: str,
        *,
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Union[ChatCompletionStreamOutput, ChatCompletionInputMessage], None]:
        self.messages.append({"role": "user", "content": user_input})

        num_turns: int = 0
        next_turn_should_call_tools = True

        while True:
            if abort_event and abort_event.is_set():
                return

            async for item in self.process_single_turn_with_tools(
                self.messages,
                exit_loop_tools=EXIT_LOOP_TOOLS,
                exit_if_first_chunk_no_tool=(num_turns > 0 and next_turn_should_call_tools),
            ):
                yield item

            num_turns += 1
            last = self.messages[-1]

            if last.get("role") == "tool" and last.get("name") in {t.function.name for t in EXIT_LOOP_TOOLS}:
                return

            if last.get("role") != "tool" and num_turns > MAX_NUM_TURNS:
                return

            if last.get("role") != "tool" and next_turn_should_call_tools:
                return

            next_turn_should_call_tools = last.get("role") != "tool"
