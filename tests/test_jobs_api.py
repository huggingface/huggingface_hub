from unittest.mock import patch

import pytest

from huggingface_hub import HfApi, JobStage
from huggingface_hub._jobs_api import (
    JobInfo,
    _default_job_name_from_image,
    _default_job_name_from_script,
    _parse_uv_job_header,
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


def _uv_script(header_lines: str) -> str:
    return f"# /// script\n# requires-python = \">=3.11\"\n{header_lines}# ///\nprint('hi')\n"


class TestParseUvJobHeader:
    def test_no_header(self) -> None:
        assert _parse_uv_job_header("print('hi')", script="x.py").is_empty()

    def test_header_without_hf_jobs_table(self) -> None:
        script = "# /// script\n# dependencies = []\n# ///\n"
        assert _parse_uv_job_header(script, script="x.py").is_empty()

    def test_full_header(self) -> None:
        script = _uv_script(
            "#\n"
            "# [tool.hf-jobs]\n"
            '# image = "vllm/vllm-openai:latest"\n'
            '# flavor = "l4x1"\n'
            '# python = "/usr/bin/python3"\n'
            '# env = { PYTHONPATH = "/usr/local/lib" }\n'
            '# secrets = ["HF_TOKEN"]\n'
            '# timeout = "30m"\n'
            '# namespace = "my-org"\n'
            '# volumes = ["hf://datasets/org/ds:/data:ro"]\n'
            '# labels = { team = "ml" }\n'
            '# name = "ocr-job"\n'
        )
        cfg = _parse_uv_job_header(script, script="ocr.py")
        assert cfg.image == "vllm/vllm-openai:latest"
        assert cfg.flavor == "l4x1"
        assert cfg.python == "/usr/bin/python3"
        assert cfg.env == {"PYTHONPATH": "/usr/local/lib"}
        assert cfg.secrets == ["HF_TOKEN"]
        assert cfg.timeout == "30m"
        assert cfg.namespace == "my-org"
        assert cfg.labels == {"team": "ml"}
        assert cfg.name == "ocr-job"
        assert not cfg.is_empty()
        volume = cfg.volumes[0]
        assert (volume.type, volume.source, volume.mount_path, volume.read_only) == (
            "dataset",
            "org/ds",
            "/data",
            True,
        )

    def test_crlf_is_normalized(self) -> None:
        script = _uv_script('# [tool.hf-jobs]\n# flavor = "l4x1"\n').replace("\n", "\r\n")
        assert _parse_uv_job_header(script, script="x.py").flavor == "l4x1"

    def test_timeout_as_int_seconds(self) -> None:
        cfg = _parse_uv_job_header(_uv_script("# [tool.hf-jobs]\n# timeout = 300\n"), script="x.py")
        assert cfg.timeout == 300

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown key.*'flavour'.*Valid keys"):
            _parse_uv_job_header(_uv_script('# [tool.hf-jobs]\n# flavour = "l4x1"\n'), script="x.py")

    @pytest.mark.parametrize(
        "line, key",
        [
            ("# flavor = 4\n", "flavor"),
            ("# env = { DEBUG = true }\n", "env"),
            ('# secrets = "HF_TOKEN"\n', "secrets"),
            ('# labels = [ "team" ]\n', "labels"),
        ],
    )
    def test_wrong_type_rejected(self, line: str, key: str) -> None:
        with pytest.raises(ValueError, match=f"Invalid `{key}` value"):
            _parse_uv_job_header(_uv_script(f"# [tool.hf-jobs]\n{line}"), script="x.py")

    def test_invalid_volume_spec_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid volume spec"):
            _parse_uv_job_header(_uv_script('# [tool.hf-jobs]\n# volumes = ["not-a-mount"]\n'), script="x.py")

    def test_invalid_toml_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid TOML"):
            _parse_uv_job_header("# /// script\n# [tool.hf-jobs\n# ///\n", script="x.py")


class TestRunUvJobHeaderConfig:
    """Header config resolution in `run_uv_job`: header values are defaults, explicit args win."""

    api = HfApi(token="hf_test")

    def _run(self, script_path, **kwargs):
        from unittest.mock import MagicMock

        with (
            patch.object(self.api, "whoami", return_value={"name": "user"}),
            patch.object(self.api, "create_bucket") as mock_bucket,
            patch.object(self.api, "batch_bucket_files"),
            patch.object(self.api, "run_job") as mock_run,
        ):
            mock_bucket.return_value = MagicMock(url="https://huggingface.co/buckets/user/jobs-artifacts")
            mock_run.return_value = MagicMock(id="job-id")
            self.api.run_uv_job(str(script_path), **kwargs)
        return mock_run.call_args.kwargs

    def test_header_values_used_as_defaults(self, tmp_path) -> None:
        script = tmp_path / "ocr.py"
        script.write_text(
            _uv_script(
                "# [tool.hf-jobs]\n"
                '# image = "vllm/vllm-openai:latest"\n'
                '# flavor = "l4x1"\n'
                '# timeout = "30m"\n'
                '# env = { PYTHONPATH = "/usr/local/lib" }\n'
                '# labels = { team = "ml" }\n'
                '# name = "ocr-job"\n'
            )
        )
        call = self._run(script)
        assert call["image"] == "vllm/vllm-openai:latest"
        assert call["flavor"] == "l4x1"
        assert call["timeout"] == "30m"
        assert call["env"] == {"PYTHONPATH": "/usr/local/lib"}
        assert call["labels"] == {"team": "ml"}
        assert call["name"] == "ocr-job"

    def test_explicit_args_override_header(self, tmp_path) -> None:
        script = tmp_path / "ocr.py"
        script.write_text(_uv_script('# [tool.hf-jobs]\n# flavor = "l4x1"\n# env = { A = "1", B = "2" }\n'))
        call = self._run(script, flavor="a10g-small", env={"B": "override", "C": "3"})
        assert call["flavor"] == "a10g-small"
        # env merges per-key: explicit wins, header-only keys kept
        assert call["env"] == {"A": "1", "B": "override", "C": "3"}

    def test_cli_label_name_overrides_header_name(self, tmp_path) -> None:
        script = tmp_path / "ocr.py"
        script.write_text(_uv_script('# [tool.hf-jobs]\n# name = "header-name"\n# labels = { team = "ml" }\n'))
        call = self._run(script, labels={"name": "cli-name"})
        assert call["labels"] == {"team": "ml", "name": "cli-name"}
        assert call["name"] is None

    def test_header_secret_resolved_from_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "secret-value")
        script = tmp_path / "job.py"
        script.write_text(_uv_script('# [tool.hf-jobs]\n# secrets = ["MY_API_KEY"]\n'))
        call = self._run(script)
        assert call["secrets"] == {"MY_API_KEY": "secret-value"}

    def test_header_hf_token_falls_back_to_saved_token(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        script = tmp_path / "job.py"
        script.write_text(_uv_script('# [tool.hf-jobs]\n# secrets = ["HF_TOKEN"]\n'))
        with patch("huggingface_hub.hf_api.get_token", return_value="hf_saved"):
            call = self._run(script)
        assert call["secrets"] == {"HF_TOKEN": "hf_saved"}

    def test_missing_header_secret_is_an_error(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("MY_API_KEY", raising=False)
        script = tmp_path / "job.py"
        script.write_text(_uv_script('# [tool.hf-jobs]\n# secrets = ["MY_API_KEY"]\n'))
        with pytest.raises(ValueError, match="requires secret.*MY_API_KEY"):
            self._run(script)

    def test_dry_run_returns_spec_without_creating_job(self, tmp_path) -> None:
        from unittest.mock import MagicMock

        script = tmp_path / "ocr.py"
        script.write_text(_uv_script('# [tool.hf-jobs]\n# flavor = "l4x1"\n# timeout = "30m"\n# name = "ocr-job"\n'))
        with (
            patch.object(self.api, "whoami", return_value={"name": "user"}),
            patch.object(self.api, "create_bucket") as mock_bucket,
            patch.object(self.api, "batch_bucket_files"),
            patch.object(self.api, "run_job") as mock_run,
        ):
            mock_bucket.return_value = MagicMock(url="https://huggingface.co/buckets/user/jobs-artifacts")
            spec = self.api.run_uv_job(str(script), _dry_run=True)
        mock_run.assert_not_called()
        assert spec["flavor"] == "l4x1"
        assert spec["timeoutSeconds"] == 1800
        assert spec["labels"] == {"name": "ocr-job"}
        assert spec["command"][:2] == ["uv", "run"]

    def test_url_script_downloaded_and_uploaded_to_bucket(self) -> None:
        from unittest.mock import MagicMock

        content = _uv_script('# [tool.hf-jobs]\n# flavor = "a10g-small"\n').encode()
        url = "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py"
        with (
            patch.object(self.api, "whoami", return_value={"name": "user"}),
            patch.object(self.api, "create_bucket") as mock_bucket,
            patch.object(self.api, "batch_bucket_files") as mock_batch,
            patch.object(self.api, "run_job") as mock_run,
            patch("huggingface_hub.hf_api.get_session") as mock_session,
        ):
            mock_bucket.return_value = MagicMock(url="https://huggingface.co/buckets/user/jobs-artifacts")
            mock_run.return_value = MagicMock(id="job-id")
            mock_session.return_value.get.return_value = MagicMock(content=content)
            self.api.run_uv_job(url, script_args=["--push_to_hub"])
        call = mock_run.call_args.kwargs
        assert call["flavor"] == "a10g-small"  # from header
        # script rewritten to the mounted bucket path, not run from the URL
        assert call["command"] == ["uv", "run", "/data/sft.py", "--push_to_hub"]
        add_ops = mock_batch.call_args.kwargs["add"]
        assert add_ops[0][0] == content  # downloaded content uploaded
        assert add_ops[0][1].endswith("/sft.py")
