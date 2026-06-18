from unittest.mock import patch

import pytest

from huggingface_hub import HfApi
from huggingface_hub._jobs_api import JobInfo


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
