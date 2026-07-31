from unittest.mock import patch

import pytest

from huggingface_hub import HfApi, JobStage
from huggingface_hub._jobs_api import (
    JobInfo,
    _default_job_name_from_image,
    _default_job_name_from_script,
)


def _job_info(stage: str, job_id: str = "job-id") -> JobInfo:
    return JobInfo(
        id=job_id,
        owner={"id": "1234", "name": "user", "type": "user"},
        status={"stage": stage},
    )


class TestWaitForJob:
    api = HfApi(token="hf_test")

    def test_polls_until_terminal_and_returns_failed_job(self) -> None:
        # A failed Job is returned, not raised: callers inspect `job.status.stage`.
        with (
            patch.object(
                self.api,
                "inspect_job",
                side_effect=[_job_info("SCHEDULING"), _job_info("RUNNING"), _job_info("ERROR")],
            ) as mock_inspect,
            patch("huggingface_hub.hf_api.time.sleep"),
        ):
            job = self.api.wait_for_job(job_id="job-id", namespace="user")
        assert job.status.stage == "ERROR"
        assert mock_inspect.call_count == 3

    def test_list_input_returns_list_in_order(self) -> None:
        with patch.object(
            self.api,
            "inspect_job",
            side_effect=lambda job_id, namespace, token: _job_info("COMPLETED", job_id=job_id),
        ):
            jobs = self.api.wait_for_job(job_id=["job-a", "job-b"], namespace="user")
        assert [job.id for job in jobs] == ["job-a", "job-b"]

    def test_raises_timeout_error(self) -> None:
        with (
            patch.object(self.api, "inspect_job", return_value=_job_info("RUNNING")),
            patch("huggingface_hub.hf_api.time.sleep"),
        ):
            with pytest.raises(TimeoutError):
                self.api.wait_for_job(job_id="job-id", timeout=0, namespace="user")

    def test_stages_waits_for_running(self) -> None:
        with (
            patch.object(
                self.api,
                "inspect_job",
                side_effect=[_job_info("SCHEDULING"), _job_info("RUNNING"), _job_info("COMPLETED")],
            ) as mock_inspect,
            patch("huggingface_hub.hf_api.time.sleep"),
        ):
            job = self.api.wait_for_job(job_id="job-id", namespace="user", stages=[JobStage.RUNNING])
        # Stops as soon as RUNNING is reached, without waiting for a terminal stage.
        assert job.status.stage == "RUNNING"
        assert mock_inspect.call_count == 2

    def test_stages_stops_on_terminal_even_if_target_not_reached(self) -> None:
        # Terminal stages always stop the wait, so waiting for RUNNING doesn't hang on a Job that fails early.
        with (
            patch.object(
                self.api,
                "inspect_job",
                side_effect=[_job_info("SCHEDULING"), _job_info("ERROR")],
            ),
            patch("huggingface_hub.hf_api.time.sleep"),
        ):
            job = self.api.wait_for_job(job_id="job-id", namespace="user", stages=[JobStage.RUNNING])
        assert job.status.stage == "ERROR"


@pytest.mark.parametrize(
    "image, expected",
    [
        # Plain image (no registry, no tag).
        ("ubuntu", "ubuntu"),
        # Tag is kept, with disallowed chars replaced by '-'.
        ("python:3.12", "python-3-12"),
        # Registry host and namespace are dropped, last component + tag is kept.
        ("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel", "pytorch-2-6-0-cuda12-4-cudnn9-devel"),
        ("ghcr.io/astral-sh/uv:python3.12-bookworm", "uv-python3-12-bookworm"),
        # Space references keep the `namespace/repo` id (sanitized), for every supported prefix.
        ("hf.co/spaces/lhoestq/duckdb", "lhoestq-duckdb"),
        ("https://huggingface.co/spaces/lhoestq/duckdb", "lhoestq-duckdb"),
    ],
)
def test_default_job_name_from_image(image: str, expected: str) -> None:
    # The base name is derived from the image; a short hash of the command line is appended.
    assert _default_job_name_from_image(image, ["python", "-c", "print(1)"]).startswith(expected + "-")


@pytest.mark.parametrize(
    "script, expected",
    [
        # Local script: keep the stem, drop the '.py' extension.
        ("my_script.py", "my_script"),
        ("./train.py", "train"),
        # URL: keep the last path component, drop query/fragment and extension.
        ("https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py", "sft"),
        ("https://example.co/a/sft.py?raw=1", "sft"),
        # Command (no extension): kept as-is.
        ("lighteval", "lighteval"),
        # Dots in the stem are disallowed chars and get replaced by '-'.
        ("my.weird.script.py", "my-weird-script"),
    ],
)
def test_default_job_name_from_script(script: str, expected: str) -> None:
    # The base name is derived from the script; a short hash of the command line is appended.
    assert _default_job_name_from_script(script, []).startswith(expected + "-")


def test_default_job_name_hash_groups_and_splits_by_command() -> None:
    # Same image but different commands must yield different names (splits distinct runs)...
    truc = _default_job_name_from_image("python:3.12", ["foo", "--truc"])
    bar = _default_job_name_from_image("python:3.12", ["foo", "--bar"])
    assert truc.startswith("python-3-12-")
    assert bar.startswith("python-3-12-")
    assert truc != bar
    # ...while the same command yields the same name (groups identical runs).
    assert truc == _default_job_name_from_image("python:3.12", ["foo", "--truc"])
