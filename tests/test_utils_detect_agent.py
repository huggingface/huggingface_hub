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
import json
import time

import pytest

from huggingface_hub import constants
from huggingface_hub.utils import _detect_agent


# A minimal registry used to keep the detection tests independent from the hardcoded fallback.
FAKE_REGISTRY = {
    "standardEnvVars": ["AI_AGENT", "AGENT"],
    "harnesses": {
        "cowork": {"envVars": {"CLAUDE_CODE_IS_COWORK": "*"}},
        "claude-code": {"envVars": {"CLAUDECODE": "*", "CLAUDE_CODE": "*"}},
        "exact-match": {"envVars": {"SOME_VAR": "expected-value"}},
        "prefix-only": {"envVars": {"PREFIX_VAR": "foo*"}},
        "devin": {},
    },
}


@pytest.fixture
def with_fake_registry(monkeypatch: pytest.MonkeyPatch):
    """Pin the in-process registry to `FAKE_REGISTRY` (no network, no disk).

    Also clears every env var referenced by `FAKE_REGISTRY` so detection is not
    influenced by agent vars that may be set on the host running the tests.
    """
    monkeypatch.setattr(_detect_agent, "_registry", FAKE_REGISTRY)
    for var in FAKE_REGISTRY["standardEnvVars"]:
        monkeypatch.delenv(var, raising=False)
    for info in FAKE_REGISTRY["harnesses"].values():
        for var in (info or {}).get("envVars", {}):
            monkeypatch.delenv(var, raising=False)
    yield


class TestDetectAgent:
    def test_no_agent(self, with_fake_registry):
        assert _detect_agent.detect_agent() is None
        assert _detect_agent.is_agent() is False

    def test_wildcard_match(self, monkeypatch, with_fake_registry):
        monkeypatch.setenv("CLAUDECODE", "1")
        assert _detect_agent.detect_agent() == "claude-code"
        assert _detect_agent.is_agent() is True

    def test_wildcard_match_ignores_empty_value(self, monkeypatch, with_fake_registry):
        monkeypatch.setenv("CLAUDECODE", "")
        assert _detect_agent.detect_agent() is None

    def test_priority_order(self, monkeypatch, with_fake_registry):
        # `cowork` is listed before `claude-code` and must win when both are set.
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_IS_COWORK", "1")
        assert _detect_agent.detect_agent() == "cowork"

    def test_exact_value_match(self, monkeypatch, with_fake_registry):
        monkeypatch.setenv("SOME_VAR", "wrong-value")
        assert _detect_agent.detect_agent() is None
        monkeypatch.setenv("SOME_VAR", "expected-value")
        assert _detect_agent.detect_agent() == "exact-match"

    def test_prefix_pattern_is_ignored(self, monkeypatch, with_fake_registry):
        # The `<prefix>*` fuzzy pattern is intentionally not implemented yet.
        monkeypatch.setenv("PREFIX_VAR", "foobar")
        assert _detect_agent.detect_agent() is None

    def test_standard_var_known_harness(self, monkeypatch, with_fake_registry):
        # `devin` has no envVars and is only detectable through the standard vars.
        monkeypatch.setenv("AGENT", "devin")
        assert _detect_agent.detect_agent() == "devin"

    def test_standard_var_is_case_insensitive(self, monkeypatch, with_fake_registry):
        monkeypatch.setenv("AI_AGENT", "Devin")
        assert _detect_agent.detect_agent() == "devin"

    def test_standard_var_unknown_harness(self, monkeypatch, with_fake_registry):
        monkeypatch.setenv("AGENT", "some-unregistered-tool")
        assert _detect_agent.detect_agent() == "unknown"

    def test_malformed_registry_with_null_values(self, monkeypatch):
        # A registry with explicit `null` values (e.g. from a malformed cache/response)
        # must not crash detection.
        monkeypatch.setattr(_detect_agent, "_registry", {"standardEnvVars": None, "harnesses": None})
        assert _detect_agent.detect_agent() is None
        assert _detect_agent.is_agent() is False


class TestRegistryLoading:
    @pytest.fixture(autouse=True)
    def reset_in_process_cache(self, monkeypatch):
        # Force `_load_registry` to run instead of returning the in-process cache.
        monkeypatch.setattr(_detect_agent, "_registry", None)
        yield

    def test_fetch_and_cache(self, monkeypatch, tmp_path):
        path = str(tmp_path / "harnesses.json")
        monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", path)
        monkeypatch.setattr(_detect_agent, "_fetch_registry", lambda: FAKE_REGISTRY)

        registry = _detect_agent._load_registry()
        assert registry == FAKE_REGISTRY
        # Persisted to disk for reuse.
        with open(path) as f:
            assert json.load(f) == FAKE_REGISTRY

    def test_fresh_cache_is_reused_without_fetching(self, monkeypatch, tmp_path):
        path = str(tmp_path / "harnesses.json")
        with open(path, "w") as f:
            json.dump(FAKE_REGISTRY, f)
        monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", path)

        def _fail():
            raise AssertionError("should not fetch when cache is fresh")

        monkeypatch.setattr(_detect_agent, "_fetch_registry", _fail)
        assert _detect_agent._load_registry() == FAKE_REGISTRY

    def test_stale_cache_triggers_refresh(self, monkeypatch, tmp_path):
        path = str(tmp_path / "harnesses.json")
        with open(path, "w") as f:
            json.dump({"standardEnvVars": [], "harnesses": {}}, f)
        # Make the file older than the TTL.
        old = time.time() - _detect_agent._REGISTRY_TTL_SECONDS - 10
        import os

        os.utime(path, (old, old))
        monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", path)
        monkeypatch.setattr(_detect_agent, "_fetch_registry", lambda: FAKE_REGISTRY)

        assert _detect_agent._load_registry() == FAKE_REGISTRY

    def test_stale_cache_reused_when_fetch_fails(self, monkeypatch, tmp_path):
        path = str(tmp_path / "harnesses.json")
        with open(path, "w") as f:
            json.dump(FAKE_REGISTRY, f)
        old = time.time() - _detect_agent._REGISTRY_TTL_SECONDS - 10
        import os

        os.utime(path, (old, old))
        monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", path)
        monkeypatch.setattr(_detect_agent, "_fetch_registry", lambda: None)

        assert _detect_agent._load_registry() == FAKE_REGISTRY

    def test_empty_when_no_cache_and_fetch_fails(self, monkeypatch, tmp_path):
        # No hardcoded fallback: an empty registry means "no agent detected".
        monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", str(tmp_path / "missing.json"))
        monkeypatch.setattr(_detect_agent, "_fetch_registry", lambda: None)
        assert _detect_agent._load_registry() == _detect_agent._EMPTY_REGISTRY

    def test_fetch_skipped_when_offline(self, monkeypatch):
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
        assert _detect_agent._fetch_registry() is None

    def test_detection_never_raises_on_error(self, monkeypatch):
        # Even if resolving the registry blows up, detection degrades gracefully to "no agent".
        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(_detect_agent, "_load_registry", _boom)
        assert _detect_agent.detect_agent() is None
        assert _detect_agent.is_agent() is False
