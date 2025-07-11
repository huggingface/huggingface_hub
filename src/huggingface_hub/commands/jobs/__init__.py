# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Contains commands to interact with jobs on the Hugging Face Hub.

Usage:
    # run a job
    huggingface-cli jobs run image command
"""

from argparse import _SubParsersAction

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.commands.jobs.cancel import CancelCommand
from huggingface_hub.commands.jobs.inspect import InspectCommand
from huggingface_hub.commands.jobs.logs import LogsCommand
from huggingface_hub.commands.jobs.ps import PsCommand
from huggingface_hub.commands.jobs.run import RunCommand
from huggingface_hub.commands.jobs.uv import UvCommand
from huggingface_hub.utils import logging


logger = logging.get_logger(__name__)


class JobsCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        jobs_parser = parser.add_parser("jobs", help="Commands to interact with your huggingface.co jobs.")
        jobs_subparsers = jobs_parser.add_subparsers(help="huggingface.co jobs related commands")

        # Register commands
        InspectCommand.register_subcommand(jobs_subparsers)
        LogsCommand.register_subcommand(jobs_subparsers)
        PsCommand.register_subcommand(jobs_subparsers)
        RunCommand.register_subcommand(jobs_subparsers)
        CancelCommand.register_subcommand(jobs_subparsers)
        UvCommand.register_subcommand(jobs_subparsers)
