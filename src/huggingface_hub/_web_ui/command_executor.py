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
"""Execute CLI commands with streaming output."""

import asyncio
import os
import re
import subprocess
from typing import AsyncGenerator


EXIT_CODE_PREFIX = "__HF_WEBUI_EXIT_CODE__:"
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


class CommandExecutor:
    """Execute HF CLI commands and stream their output."""

    @staticmethod
    async def execute_command(command: list[str]) -> AsyncGenerator[str, None]:
        """
        Execute a CLI command and stream output in real-time.

        Args:
            command: List of command arguments (e.g., ['hf', 'repo', 'ls', 'username/repo'])

        Yields:
            Output lines from the command
        """
        process = None
        try:
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            env["CLICOLOR"] = "0"

            # Use subprocess to run the command
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )

            # Stream output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield _strip_ansi(line.decode("utf-8", errors="replace"))

            # Wait for process to complete
            returncode = await process.wait()
            yield f"{EXIT_CODE_PREFIX}{returncode}"

        except Exception as e:
            yield f"Error executing command: {str(e)}"
            yield f"{EXIT_CODE_PREFIX}1"
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
