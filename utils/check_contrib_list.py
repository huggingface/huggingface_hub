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
"""Contains a tool to list contrib test suites automatically."""

import argparse
import re
from pathlib import Path
from typing import NoReturn


ROOT_DIR = Path(__file__).parent.parent
CONTRIB_PATH = ROOT_DIR / "contrib"
MAKEFILE_PATH = ROOT_DIR / "Makefile"
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "contrib-tests.yml"

MAKEFILE_REGEX = re.compile(r"^CONTRIB_LIBS := .*$", flags=re.MULTILINE)
WORKFLOW_REGEX = re.compile(
    r"""
    # First: match "contrib: ["
    (?P<before>^\s{8}contrib:\s\[\n)
    # Match list of libs
    (\s{10}\".*\",\n)*
    # Finally: match trailing "]"
    (?P<after>^\s{8}\])
    """,
    flags=re.MULTILINE | re.VERBOSE,
)


def check_contrib_list(update: bool) -> NoReturn:
    """List `contrib` test suites.

    Make sure `Makefile` and `.github/workflows/contrib-tests.yml` are consistent with
    the list."""
    # List contrib test suites
    contrib_list = sorted(
        path.name for path in CONTRIB_PATH.glob("*") if path.is_dir() and not path.name.startswith("_")
    )

    # Check Makefile is consistent with list
    makefile_content = MAKEFILE_PATH.read_text()
    makefile_expected_content = MAKEFILE_REGEX.sub(f"CONTRIB_LIBS := {' '.join(contrib_list)}", makefile_content)

    # Check workflow is consistent with list
    workflow_content = WORKFLOW_PATH.read_text()
    _substitute = "\n".join(f'{" " * 10}"{lib}",' for lib in contrib_list)
    workflow_content_expected = WORKFLOW_REGEX.sub(rf"\g<before>{_substitute}\n\g<after>", workflow_content)

    #
    failed = False
    if makefile_content != makefile_expected_content:
        if update:
            print(
                "✅ Contrib libs have been updated in `Makefile`."
                "\n   Please make sure the changes are accurate and commit them."
            )
            MAKEFILE_PATH.write_text(makefile_expected_content)
        else:
            print(
                "❌ Expected content mismatch in `Makefile`.\n   It is most likely that"
                " you added a contrib test and did not update the Makefile.\n   Please"
                " run `make style` or `python utils/check_contrib_list.py --update`."
            )
            failed = True

    if workflow_content != workflow_content_expected:
        if update:
            print(
                f"✅ Contrib libs have been updated in `{WORKFLOW_PATH}`."
                "\n   Please make sure the changes are accurate and commit them."
            )
            WORKFLOW_PATH.write_text(workflow_content_expected)
        else:
            print(
                f"❌ Expected content mismatch in `{WORKFLOW_PATH}`.\n   It is most"
                " likely that you added a contrib test and did not update the github"
                " workflow file.\n   Please run `make style` or `python"
                " utils/check_contrib_list.py --update`."
            )
            failed = True

    if failed:
        exit(1)
    print("✅ All good! (contrib list)")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Whether to fix Makefile and github workflow if a new lib is detected.",
    )
    args = parser.parse_args()

    check_contrib_list(update=args.update)
