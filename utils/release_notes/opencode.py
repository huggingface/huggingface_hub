#!/usr/bin/env python3
"""Shared OpenCode helpers for the release notes scripts."""

import shutil
import subprocess


def check_opencode_model(model: str) -> None:
    """Verify that ``model`` is listed by ``opencode models``.

    OpenCode exits 0 and prints an error line when an unknown model is passed
    via ``--model``, so calls silently no-op unless we validate up front.
    Raises ``RuntimeError`` if opencode is missing or the model is unknown.

    ``--refresh`` is what makes this reliable on CI. OpenCode reads its model
    catalog from ``~/.cache/opencode/models.json`` and falls back to the
    models.dev snapshot embedded in the binary at build time when that file is
    missing; the refresh it triggers runs in a background fiber that is never
    awaited. On a fresh runner the first call therefore lists the snapshot of
    the pinned OpenCode version, so any model released after that version looks
    unknown. ``--refresh`` fetches the catalog up front and warms the cache for
    the ``opencode run`` calls that follow.
    """
    opencode_cmd = shutil.which("opencode")
    if not opencode_cmd:
        raise RuntimeError("'opencode' command not found in PATH")

    result = subprocess.run([opencode_cmd, "models", "--refresh"], check=True, capture_output=True, text=True)
    available = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if model not in available:
        raise RuntimeError(
            f"RELEASE_NOTES_MODEL={model!r} not found in `opencode models` output. "
            f"Expected the full `provider/model` form (e.g. `huggingface/zai-org/GLM-4.6`). "
            f"{len(available)} model(s) available — first 10: {', '.join(available[:10])}"
        )
