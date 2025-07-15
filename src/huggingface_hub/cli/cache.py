# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Contains commands for cache management.

Usage:
    hf cache scan
    hf cache delete
"""

from argparse import _SubParsersAction

from huggingface_hub.cli import BaseHfCLICommand
from huggingface_hub.commands.delete_cache import DeleteCacheCommand as OldDeleteCacheCommand
from huggingface_hub.commands.scan_cache import ScanCacheCommand as OldScanCacheCommand


class CacheCommand(BaseHfCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        cache_parser = parser.add_parser("cache", help="Cache commands")
        cache_subparsers = cache_parser.add_subparsers(help="Cache subcommands")

        # Scan command
        scan_parser = cache_subparsers.add_parser("scan", help="Scan cache directory")
        scan_parser.add_argument("--dir", type=str, help="Cache directory to scan")
        scan_parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbose mode")
        scan_parser.set_defaults(func=lambda args: CacheScanCommand(args))

        # Delete command
        delete_parser = cache_subparsers.add_parser("delete", help="Delete revisions from the cache directory")
        delete_parser.add_argument("--dir", type=str, help="Cache directory to clean")
        delete_parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbose mode")
        delete_parser.add_argument("--disable-tui", action="store_true", help="Disable TUI")
        delete_parser.add_argument("-y", "--yes", action="store_true", help="Assume yes to all prompts")
        delete_parser.set_defaults(func=lambda args: CacheDeleteCommand(args))

    def run(self):
        # This should never be called since we have subcommands
        raise NotImplementedError("Cache command requires a subcommand")


class CacheScanCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent CacheCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldScanCacheCommand(self.args)
        old_command.run()


class CacheDeleteCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent CacheCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldDeleteCacheCommand(self.args)
        old_command.run()