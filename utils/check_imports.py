# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Verify the public API imports cleanly, including from a cold interpreter.

Run by the `check-imports` CI job (which installs base dependencies only) to ensure:

1. every public name is importable — catches a newly added top-level import of an optional dependency;
2. each lazily-loaded submodule imports on its own in a *fresh* interpreter.

(2) matters because `from huggingface_hub import *` warms up `sys.modules`, which can mask circular
imports that only surface on a cold-start import order (see
https://github.com/huggingface/huggingface_hub/issues/4384). Importing one representative name per
submodule in a separate process recreates that cold start — driven from `_SUBMOD_ATTRS`, so no module
list has to be maintained by hand.
"""

import subprocess
import sys

import huggingface_hub


def _import(statement: str) -> tuple[bool, str]:
    """Run `statement` in a fresh interpreter; return (succeeded, last stderr line)."""
    result = subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True)
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip()
    return False, stderr.splitlines()[-1] if stderr else f"exited with code {result.returncode}"


def main() -> int:
    # `import *` resolves every public name in one (warm) process. It cannot catch import-ordering /
    # circular-import bugs though, because it warms up `sys.modules` first: so additionally import
    # each lazily-loaded submodule on its own, in a fresh interpreter, where the cold path is exposed.
    statements = ["from huggingface_hub import *"]
    statements += [f"import huggingface_hub.{submodule}" for submodule in huggingface_hub._SUBMOD_ATTRS]

    failures = []
    for statement in statements:
        ok, error = _import(statement)
        print(f"  {'ok  ' if ok else 'FAIL'}  {statement}")
        if not ok:
            failures.append((statement, error))

    if failures:
        print("\nImport failures (circular import or missing dependency):", file=sys.stderr)
        for statement, error in failures:
            print(f"  - {statement}\n      {error}", file=sys.stderr)
        return 1
    print(f"\nAll {len(statements)} imports succeed from a cold interpreter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
