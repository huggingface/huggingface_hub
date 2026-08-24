import io
import logging
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from tqdm.auto import tqdm as vanilla_tqdm

from huggingface_hub.utils import (
    SoftTemporaryDirectory,
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
    hf_thread_map,
    tqdm,
    tqdm_stream_file,
)
from huggingface_hub.utils.tqdm import _get_progress_bar_context


_ENV_VAR = "huggingface_hub.utils._tqdm.HF_HUB_DISABLE_PROGRESS_BARS"


@pytest.fixture(autouse=True)
def _reset_progress_bars():
    enable_progress_bars()
    yield
    enable_progress_bars()


@pytest.fixture(autouse=True)
def _no_env_override():
    with patch(_ENV_VAR, None):
        yield


@pytest.fixture()
def _env_force_disabled():
    with patch(_ENV_VAR, True):
        yield


@pytest.fixture()
def _env_force_enabled():
    with patch(_ENV_VAR, False):
        yield


class TestTqdmUtils:
    def test_tqdm_helpers(self):
        disable_progress_bars()
        assert are_progress_bars_disabled()

        enable_progress_bars()
        assert not are_progress_bars_disabled()

    def test_cannot_enable_when_env_disables(self, _env_force_disabled):
        disable_progress_bars()
        assert are_progress_bars_disabled()

        with pytest.warns(UserWarning):
            enable_progress_bars()
        assert are_progress_bars_disabled()

    def test_cannot_disable_when_env_enables(self, _env_force_enabled):
        enable_progress_bars()
        assert not are_progress_bars_disabled()

        with pytest.warns(UserWarning):
            with disable_progress_bars():
                pass
        assert not are_progress_bars_disabled()

    def test_tqdm_disabled(self, capsys):
        disable_progress_bars()
        for _ in tqdm(range(10)):
            pass

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_tqdm_disabled_cannot_be_forced(self, capsys):
        disable_progress_bars()
        for _ in tqdm(range(10), disable=False):
            pass

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_tqdm_can_be_disabled_when_globally_enabled(self, capsys):
        enable_progress_bars()
        for _ in tqdm(range(10), disable=True):
            pass

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_tqdm_enabled(self, capsys):
        enable_progress_bars()
        for _ in tqdm(range(10)):
            pass

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "10/10" in captured.err

    def test_tqdm_stream_file(self, capsys):
        enable_progress_bars()
        with SoftTemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"
            with filepath.open("w") as f:
                f.write("#" * 1000)

            with tqdm_stream_file(filepath) as f:
                while True:
                    data = f.read(100)
                    if not data:
                        break
                    time.sleep(0.001)

            captured = capsys.readouterr()
            assert captured.out == ""
            assert "config.json: 100%" in captured.err
            assert "|█████████" in captured.err
            assert "1.00k/1.00k" in captured.err


class TestTqdmGroup:
    def test_disable_specific_group(self):
        disable_progress_bars("peft.foo")
        assert not are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.something")
        assert are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo.bar")

    def test_enable_specific_subgroup(self):
        disable_progress_bars("peft.foo")
        enable_progress_bars("peft.foo.bar")
        assert are_progress_bars_disabled("peft.foo")
        assert not are_progress_bars_disabled("peft.foo.bar")

    def test_disable_override_by_environment_variable(self, _env_force_disabled):
        with pytest.warns(UserWarning):
            enable_progress_bars()
        assert are_progress_bars_disabled("peft")
        assert are_progress_bars_disabled("peft.foo")

    def test_enable_override_by_environment_variable(self, _env_force_enabled):
        with pytest.warns(UserWarning):
            disable_progress_bars("peft.foo")
        assert not are_progress_bars_disabled("peft.foo")

    def test_partial_group_name_not_affected(self):
        disable_progress_bars("peft.foo")
        assert not are_progress_bars_disabled("peft.footprint")

    def test_nested_subgroup_behavior(self):
        disable_progress_bars("peft")
        enable_progress_bars("peft.foo")
        disable_progress_bars("peft.foo.bar")
        assert are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo.bar")

    def test_empty_group_is_root(self):
        disable_progress_bars("")
        assert not are_progress_bars_disabled("peft")

        enable_progress_bars("123.invalid.name")
        assert not are_progress_bars_disabled("123.invalid.name")

    def test_multiple_level_toggling(self):
        disable_progress_bars("peft")
        enable_progress_bars("peft.foo")
        disable_progress_bars("peft.foo.bar.something")
        assert are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo.bar.something")

    def test_progress_bar_respects_group(self, capsys):
        disable_progress_bars("foo.bar")
        for _ in tqdm(range(10), name="foo.bar.something"):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        enable_progress_bars("foo.bar.something")
        for _ in tqdm(range(10), name="foo.bar.something"):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "10/10" in captured.err

    def test_progress_bar_context_respects_group(self):
        disable_progress_bars("foo.bar")
        with _get_progress_bar_context(
            desc="test",
            log_level=logging.INFO,
            total=10,
            name="foo.bar.something",
        ) as pbar:
            assert pbar.disable


class TestDisableProgressBarsContextManager:
    def test_restores_state(self):
        assert not are_progress_bars_disabled()
        with disable_progress_bars():
            assert are_progress_bars_disabled()
        assert not are_progress_bars_disabled()

    def test_with_group(self):
        with disable_progress_bars("peft.foo"):
            assert are_progress_bars_disabled("peft.foo")
            assert are_progress_bars_disabled("peft.foo.bar")
            assert not are_progress_bars_disabled("peft")
        assert not are_progress_bars_disabled("peft.foo")

    def test_noop_when_already_disabled(self):
        disable_progress_bars()
        assert are_progress_bars_disabled()
        with disable_progress_bars():
            assert are_progress_bars_disabled()
        assert are_progress_bars_disabled()

    def test_noop_when_group_already_disabled(self):
        disable_progress_bars("peft.foo")
        with disable_progress_bars("peft.foo"):
            assert are_progress_bars_disabled("peft.foo")
        assert are_progress_bars_disabled("peft.foo")

    def test_nested(self):
        assert not are_progress_bars_disabled()
        with disable_progress_bars():
            assert are_progress_bars_disabled()
            with disable_progress_bars():
                assert are_progress_bars_disabled()
            assert are_progress_bars_disabled()
        assert not are_progress_bars_disabled()

    def test_suppresses_output(self, capsys):
        with disable_progress_bars():
            for _ in tqdm(range(10)):
                pass
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

        for _ in tqdm(range(10)):
            pass
        captured = capsys.readouterr()
        assert "10/10" in captured.err


class TestHfThreadMap:
    def test_returns_results_in_input_order(self):
        # Earlier items finish later, so completion order differs from input order.
        def fn(x):
            time.sleep(0.02 * (4 - x))
            return x * 10

        assert hf_thread_map(fn, range(5), max_workers=5, disable=True) == [0, 10, 20, 30, 40]

    def test_bar_advances_on_completion_not_submission_order(self):
        # Regression test for https://github.com/huggingface/huggingface_hub/issues/4518: a slow
        # first-submitted item must not prevent the bar from advancing when later items complete.
        slow_release = threading.Event()
        fast_done = threading.Event()
        updates = []

        class RecordingTqdm(vanilla_tqdm):
            def update(self, n=1):
                updates.append(n)
                return super().update(n)

        def fn(x):
            if x == 0:
                slow_release.wait(timeout=5)  # first-submitted item stays blocked
            else:
                fast_done.set()
            return x

        results = []

        def run():
            results.append(hf_thread_map(fn, range(2), max_workers=2, tqdm_class=RecordingTqdm, disable=True))

        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert fast_done.wait(timeout=5)
            # The fast item completed; the bar should advance even though the slow item is still blocked.
            deadline = time.time() + 5
            while not updates and time.time() < deadline:
                time.sleep(0.01)
            assert updates
        finally:
            slow_release.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert results == [[0, 1]]

    def test_cancels_pending_tasks_on_error(self):
        started = []
        lock = threading.Lock()

        def fn(x):
            with lock:
                started.append(x)
            if x == 0:
                raise ValueError("boom")
            time.sleep(0.05)
            return x

        with pytest.raises(ValueError, match="boom"):
            hf_thread_map(fn, range(50), max_workers=1, disable=True)

        # The failing first task cancels the queued tasks instead of running all 50.
        assert len(started) < 50

    def test_shares_tqdm_lock_with_worker_threads(self):
        # A custom bar class: hf_thread_map must share a single lock with the worker threads so bars
        # created inside `fn` don't race on concurrent updates (mirrors tqdm.contrib's `ensure_lock`).
        class CustomTqdm(vanilla_tqdm):
            pass

        seen_locks = []
        lock = threading.Lock()

        def fn(x):
            with lock:
                seen_locks.append(CustomTqdm.get_lock())
            return x

        hf_thread_map(fn, range(8), max_workers=4, tqdm_class=CustomTqdm, disable=True)

        # Every worker thread observed the same, single shared lock.
        assert seen_locks
        assert all(observed is seen_locks[0] for observed in seen_locks)

    def test_works_with_class_without_lock_api(self):
        # Classes that don't implement tqdm's locking API (e.g. `silent_tqdm`) must not crash.
        from huggingface_hub.utils import silent_tqdm

        assert hf_thread_map(lambda x: x * 2, range(4), max_workers=2, tqdm_class=silent_tqdm) == [0, 2, 4, 6]


class TestCreateProgressBarCustomClass:
    def test_custom_class_not_disabled_in_non_tty(self):
        fake_stderr = io.StringIO()
        with patch.object(sys, "stderr", fake_stderr):
            with _get_progress_bar_context(
                desc="test",
                log_level=logging.INFO,
                total=100,
                tqdm_class=vanilla_tqdm,
                name="huggingface_hub.test",
            ) as pbar:
                assert not pbar.disable
                pbar.update(50)
                pbar.update(50)
                assert pbar.n == 100

    def test_custom_class_ignores_hf_disable_signal(self, _env_force_disabled):
        with _get_progress_bar_context(
            desc="test",
            log_level=logging.INFO,
            total=10,
            tqdm_class=vanilla_tqdm,
            name="huggingface_hub.test",
        ) as pbar:
            assert not pbar.disable

    def test_custom_class_no_name_kwarg(self):
        with _get_progress_bar_context(
            desc="test",
            log_level=logging.INFO,
            total=10,
            tqdm_class=vanilla_tqdm,
            name="huggingface_hub.test",
        ) as pbar:
            pbar.update(10)
