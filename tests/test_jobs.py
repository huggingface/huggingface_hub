from unittest.mock import patch

import pytest

from huggingface_hub import HfApi
from huggingface_hub._jobs_api import JobInfo
from huggingface_hub.errors import JobTimeoutError


def _job(stage: str) -> JobInfo:
    return JobInfo(id="job-id", status={"stage": stage}, owner={"id": "i", "name": "me", "type": "user"})


class TestWaitForJob:
    def test_returns_when_completed(self) -> None:
        api = HfApi()
        with (
            patch.object(api, "whoami", return_value={"name": "me"}) as whoami,
            patch.object(
                api, "inspect_job", side_effect=[_job("RUNNING"), _job("RUNNING"), _job("COMPLETED")]
            ) as inspect,
            patch("huggingface_hub.hf_api.time.sleep"),
        ):
            job = api.wait_for_job(job_id="job-id")
        assert job.status.stage == "COMPLETED"
        assert inspect.call_count == 3
        # Namespace is resolved once up front, not on every poll.
        whoami.assert_called_once()
        inspect.assert_called_with(job_id="job-id", namespace="me", token=None)

    def test_returns_terminal_failure_without_raising(self) -> None:
        api = HfApi()
        with (
            patch.object(api, "inspect_job", return_value=_job("ERROR")),
            patch("huggingface_hub.hf_api.time.sleep"),
        ):
            job = api.wait_for_job(job_id="job-id", namespace="me")
        # A failed Job is a normal terminal outcome: it is returned, not raised.
        assert job.status.stage == "ERROR"

    def test_raises_on_timeout(self) -> None:
        api = HfApi()
        with (
            patch.object(api, "inspect_job", return_value=_job("RUNNING")),
            patch("huggingface_hub.hf_api.time.sleep"),
            patch("huggingface_hub.hf_api.time.time", side_effect=[0.0, 100.0]),
        ):
            with pytest.raises(JobTimeoutError):
                api.wait_for_job(job_id="job-id", namespace="me", timeout=10)

    def test_invalid_arguments(self) -> None:
        api = HfApi()
        with pytest.raises(ValueError):
            api.wait_for_job(job_id="job-id", namespace="me", timeout=-1)
        with pytest.raises(ValueError):
            api.wait_for_job(job_id="job-id", namespace="me", refresh_every=0)

    def test_jobinfo_wait_delegates_to_api(self) -> None:
        job = _job("RUNNING")
        with patch("huggingface_hub.hf_api.HfApi.wait_for_job", return_value=_job("COMPLETED")) as wait_for_job:
            result = job.wait(timeout=30)
        assert result.status.stage == "COMPLETED"
        wait_for_job.assert_called_once_with(job_id="job-id", namespace="me", timeout=30, refresh_every=5, token=None)
