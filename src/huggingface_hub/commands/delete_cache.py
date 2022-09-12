# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains command to delete some revisions from the HF cache directory.

Usage:
    huggingface-cli delete-cache
    huggingface-cli delete-cache --keep-last
    huggingface-cli delete-cache --keep-last -y

NOTE:
    This command is based on `InquirerPy` to build the multiselect menu in the terminal.
    This dependency has to be installed with `pip install huggingface_hub[cli]`. Since
    we want to avoid as much as possible cross-platform issues, I chose a library that
    is built on top of `python-prompt-toolkit` which seems to be a reference in terminal
    GUI (actively maintained on both Unix and Windows, 7.9k stars).

    For the moment, the TUI feature is in beta.

    See:
    - https://github.com/kazhala/InquirerPy
    - https://inquirerpy.readthedocs.io/en/latest/
    - https://github.com/prompt-toolkit/python-prompt-toolkit

    Other solutions could have been:
    - `simple_term_menu`: would be good as well for our use case but some issues suggest
      that Windows is less supported.
      See: https://github.com/IngoMeyer441/simple-term-menu
    - `PyInquirer`: very similar to `InquirerPy` but older and not maintained anymore.
      In particular, no support of Python3.10.
      See: https://github.com/CITGuru/PyInquirer
    - `pick` (or `pickpack`): easy to use and flexible but built on top of Python's
      standard library `curses` that is specific to Unix (not implemented on Windows).
      See https://github.com/wong2/pick and https://github.com/anafvana/pickpack.
    - `inquirer`: lot of traction (700 stars) but explicitly states "experimental
      support of Windows". Not built on top of `python-prompt-toolkit`.
      See https://github.com/magmax/python-inquirer
"""
from argparse import ArgumentParser
from typing import Iterable, List, Optional

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from ..utils import CachedRepoInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand


# Possibility for the user to cancel deletion
_NO_DELETION_STR = "NO_DELETION_NOT_A_REVISION"


class DeleteCacheCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        delete_cache_parser = parser.add_parser(
            "delete-cache", help="Delete revisions from the cache directory."
        )

        delete_cache_parser.add_argument(
            "--dir",
            type=str,
            default=None,
            help=(
                "cache directory (optional). Default to the default HuggingFace cache."
            ),
        )
        delete_cache_parser.set_defaults(func=DeleteCacheCommand)

    def __init__(self, args):
        self.cache_dir: Optional[str] = args.dir

    def run(self):
        # TODO: add support for `huggingface-cli delete-cache aaaaaa bbbbbb cccccc (...)` as pre-selection
        # TODO: add "--keep-last" arg to delete revisions that are not on `main` ref
        # TODO: add "--filter" arg to filter repositories by name
        # TODO: add "--dry-run" and "--from-dry-run" to bypass the TUI
        # TODO: add "--sort" arg to sort by size ?
        # TODO: add "--limit" arg to limit to X repos ?
        # TODO: add "-y" arg for immediate deletion ?
        # See https://github.com/huggingface/huggingface_hub/issues/1025

        # Scan cache directory
        hf_cache_info = scan_cache_dir(self.cache_dir)

        # Manual review from the user
        selected_hashes = self._manual_review(hf_cache_info, preselected_hashes=[])

        # If deletion is not cancelled
        if len(selected_hashes) > 0 and _NO_DELETION_STR not in selected_hashes:
            confirmed = inquirer.confirm(
                self._get_instructions_str(hf_cache_info, selected_hashes)
                + " Confirm deletion ?",
                default=True,
            ).execute()

            # Deletion is confirmed
            if confirmed:
                strategy = hf_cache_info.delete_revisions(*selected_hashes)
                print("Start deletion.")
                strategy.execute()
                print(
                    f"Done. Deleted {len(strategy.repos)} repo(s) and"
                    f" {len(strategy.snapshots)} revision(s) for a total of"
                    f" {strategy.expected_freed_size_str}."
                )
                return

        # Deletion is cancelled
        print("Deletion is cancelled. Do nothing.")

    def _manual_review(
        self, hf_cache_info: HFCacheInfo, preselected_hashes: List[str]
    ) -> List[str]:
        """Displays a multi-select menu in the terminal for the user to manually select
        and review the revisions to be deleted.

        Some revisions can be preselected.
        """
        # Define multiselect list
        choices = self._get_choices_from_scan(
            repos=hf_cache_info.repos, preselected_hashes=preselected_hashes
        )
        checkbox = inquirer.checkbox(
            message="Select revisions to delete:",
            choices=choices,  # List of revisions with some pre-selection
            cycle=False,  # No loop between top and bottom
            height=16,  # Large list is possible
            # We use the instruction to display to the user the expected effect of the
            # deletion.
            instruction=self._get_instructions_str(
                hf_cache_info,
                selected_hashes=[
                    c.value for c in choices if isinstance(c, Choice) and c.enabled
                ],
            ),
            # We use the long instruction to should keybindings instructions to the user
            long_instruction=(
                "Press <space> to select, <enter> to validate and <ctrl+c> to quit"
                " without modification."
            ),
            # Message that is displayed once the user validates its selection.
            transformer=lambda result: f"{len(result)} revision(s) selected.",
        )

        # Add a callback to update the information line when a revision is
        # selected/unselected
        def _update_expectations(_) -> None:
            # Hacky way to dynamically set an instruction message to the checkbox when
            # a revision hash is selected/unselected.
            checkbox._instruction = self._get_instructions_str(
                hf_cache_info,
                selected_hashes=[
                    choice["value"]
                    for choice in checkbox.content_control.choices
                    if choice["enabled"]
                ],
            )

        checkbox.kb_func_lookup["toggle"].append({"func": _update_expectations})

        # Finally display the form to the user.
        try:
            return checkbox.execute()
        except KeyboardInterrupt:
            return []  # Quit without deletion

    def _get_choices_from_scan(
        self, repos: Iterable[CachedRepoInfo], preselected_hashes: List[str]
    ) -> List:
        """Build a list of choices from the scanned repos.

        Args:
            repos (*Iterable[`CachedRepoInfo`]*):
                List of scanned repos on which we want to delete revisions.
            preselected_hashes (*List[`str`]*):
                List of revision hashes that will be preselected.

        Return:
            The list of choices to pass to `inquirer.checkbox`.
        """
        choices = []

        # First choice is to cancel the deletion. If selected, nothing will be deleted,
        # no matter the other selected items.
        choices.append(
            Choice(
                _NO_DELETION_STR,
                name="None of the following (if selected, nothing will be deleted).",
                enabled=False,
            )
        )

        # Display a separator per repo and a Choice for each revisions of the repo
        for repo in sorted(repos, key=lambda r: (r.repo_type, r.repo_id)):
            # Repo as separator
            choices.append(
                Separator(
                    f"\n{repo.repo_type.capitalize()} {repo.repo_id} ({repo.size_on_disk_str},"
                    f" used {repo.last_accessed_str})"
                )
            )
            for revision in sorted(repo.revisions, key=lambda r: r.commit_hash):
                # Revision as choice
                choices.append(
                    Choice(
                        revision.commit_hash,
                        name=(
                            f"{revision.commit_hash[:8]}:"
                            f" {', '.join(revision.refs or 'detached')} # modified"
                            f" {revision.last_modified_str}"
                        ),
                        enabled=revision.commit_hash in preselected_hashes,
                    )
                )

        # Return choices
        return choices

    def _get_instructions_str(
        self, hf_cache_info: HFCacheInfo, selected_hashes: List[str]
    ) -> str:
        if _NO_DELETION_STR in selected_hashes:
            return "Nothing will be deleted."
        strategy = hf_cache_info.delete_revisions(*selected_hashes)
        return (
            f"{len(selected_hashes)} revisions selected counting for"
            f" {strategy.expected_freed_size_str}."
        )
