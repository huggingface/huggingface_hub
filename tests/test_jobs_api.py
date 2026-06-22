from unittest.mock import Mock, patch

import pytest

from huggingface_hub import HfApi, JobStage
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


class TestListJobs:
    api = HfApi(token="hf_test")

    def _patch_session(self):
        response = Mock()
        response.json.return_value = []
        session = patch("huggingface_hub.hf_api.get_session")
        return session, response

    def test_forwards_status_and_labels_as_query_params(self) -> None:
        session, response = self._patch_session()
        with session as mock_session:
            mock_session.return_value.get.return_value = response
            self.api.list_jobs(
                namespace="user",
                status=[JobStage.RUNNING, "scheduling"],
                labels={"env": "prod", "team": "ml"},
            )
        params = mock_session.return_value.get.call_args.kwargs["params"]
        # `status` is forwarded as the `stage` query param, upper-cased; labels as `key=value`.
        assert params == [
            ("stage", "RUNNING"),
            ("stage", "SCHEDULING"),
            ("label", "env=prod"),
            ("label", "team=ml"),
        ]

    def test_single_status_string_is_wrapped(self) -> None:
        session, response = self._patch_session()
        with session as mock_session:
            mock_session.return_value.get.return_value = response
            self.api.list_jobs(namespace="user", status="RUNNING")
        assert mock_session.return_value.get.call_args.kwargs["params"] == [("stage", "RUNNING")]

    def test_no_filters_sends_no_params(self) -> None:
        session, response = self._patch_session()
        with session as mock_session:
            mock_session.return_value.get.return_value = response
            self.api.list_jobs(namespace="user")
        assert mock_session.return_value.get.call_args.kwargs["params"] is None
