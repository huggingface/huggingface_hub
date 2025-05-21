from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Sequence, Union

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient

from .._providers import PROVIDER_OR_POLICY_T
from .constants import DEFAULT_SYSTEM_PROMPT, EXIT_LOOP_TOOLS, MAX_NUM_TURNS


class Agent(MCPClient):
    """
    Python implementation of the huggingface/mcp-client JS tiny agent
    """

    def __init__(
        self,
        *,
        model: str,
        servers: Sequence[Dict],
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
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
            await self.add_mcp_server(**cfg)

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
