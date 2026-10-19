from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import hf_hub_download


def test_existing_local_file_survives_failed_download(tmp_path: Path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    file_path = local_dir / "model.bin"
    original_content = b"original model data"
    file_path.write_bytes(original_content)

    # Simulate a successful metadata lookup followed by a failed download.
    with (
        patch(
            "huggingface_hub.file_download._get_metadata_or_catch_error",
            return_value=(
                "a" * 40,
                "new-etag",
                "https://huggingface.co/test/repo/resolve/main/model.bin",
                100,
                None,
                None,
            ),
        ),
        patch(
            "huggingface_hub.file_download._download_to_tmp_and_move",
            side_effect=OSError("Simulated download failure"),
        ),
    ):
        with pytest.raises(OSError, match="Simulated download failure"):
            hf_hub_download(
                repo_id="test/repo",
                filename="model.bin",
                local_dir=local_dir,
                force_download=True,
            )

    assert file_path.exists(), "Existing local file was deleted"
    assert file_path.read_bytes() == original_content


def test_existing_local_file_is_replaced_after_successful_download(tmp_path: Path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    file_path = local_dir / "model.bin"
    file_path.write_bytes(b"original model data")

    new_content = b"updated model data"

    def mock_download(*args, **kwargs):
        destination = kwargs["destination_path"]
        destination.write_bytes(new_content)

    with (
        patch(
            "huggingface_hub.file_download._get_metadata_or_catch_error",
            return_value=(
                "a" * 40,
                "new-etag",
                "https://huggingface.co/test/repo/resolve/main/model.bin",
                len(new_content),
                None,
                None,
            ),
        ),
        patch(
            "huggingface_hub.file_download._download_to_tmp_and_move",
            side_effect=mock_download,
        ) as download_mock,
    ):
        result = hf_hub_download(
            repo_id="test/repo",
            filename="model.bin",
            local_dir=local_dir,
            force_download=False,
        )

    assert Path(result) == file_path
    assert file_path.read_bytes() == new_content
    download_mock.assert_called_once()
    assert download_mock.call_args.kwargs["force_download"] is True


def test_download_replaces_existing_file_only_after_completion(tmp_path: Path):
    from huggingface_hub.file_download import _download_to_tmp_and_move

    destination = tmp_path / "model.bin"
    destination.write_bytes(b"original model data")

    incomplete = tmp_path / "model.incomplete"
    new_content = b"updated model data"

    def mock_http_get(url, temp_file, **kwargs):
        # The original file must still exist during the download.
        assert destination.read_bytes() == b"original model data"
        temp_file.write(new_content)

    with patch(
        "huggingface_hub.file_download.http_get",
        side_effect=mock_http_get,
    ):
        _download_to_tmp_and_move(
            incomplete_path=incomplete,
            destination_path=destination,
            url_to_download="https://example.com/model.bin",
            headers={},
            expected_size=len(new_content),
            filename="model.bin",
            force_download=True,
            etag="new-etag",
            xet_file_data=None,
        )

    assert destination.read_bytes() == new_content
    assert not list(tmp_path.glob("*.incomplete"))


def test_existing_local_file_survives_failed_cache_copy(tmp_path: Path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    destination = local_dir / "model.bin"
    original_content = b"original model data"
    destination.write_bytes(original_content)

    cached_file = tmp_path / "cached-model.bin"
    cached_file.write_bytes(b"updated model data")

    def failing_copy(source, target, *args, **kwargs):
        # Simulate a copy that modifies the destination before failing.
        Path(target).write_bytes(b"partial data")
        raise OSError("Simulated cache copy failure")

    with (
        patch(
            "huggingface_hub.file_download._get_metadata_or_catch_error",
            return_value=(
                "a" * 40,
                "new-etag",
                "https://huggingface.co/test/repo/resolve/main/model.bin",
                len(cached_file.read_bytes()),
                None,
                None,
            ),
        ),
        patch(
            "huggingface_hub.file_download.try_to_load_from_cache",
            return_value=str(cached_file),
        ),
        patch(
            "huggingface_hub.file_download.shutil.copyfile",
            side_effect=failing_copy,
        ),
    ):
        with pytest.raises(OSError, match="Simulated cache copy failure"):
            hf_hub_download(
                repo_id="test/repo",
                filename="model.bin",
                local_dir=local_dir,
                force_download=False,
            )

    assert destination.read_bytes() == original_content


def test_existing_local_file_is_replaced_from_cache(tmp_path: Path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    destination = local_dir / "model.bin"
    destination.write_bytes(b"original model data")

    cached_file = tmp_path / "cached-model.bin"
    new_content = b"updated model data"
    cached_file.write_bytes(new_content)

    with (
        patch(
            "huggingface_hub.file_download._get_metadata_or_catch_error",
            return_value=(
                "a" * 40,
                "new-etag",
                "https://huggingface.co/test/repo/resolve/main/model.bin",
                len(new_content),
                None,
                None,
            ),
        ),
        patch(
            "huggingface_hub.file_download.try_to_load_from_cache",
            return_value=str(cached_file),
        ),
        patch(
            "huggingface_hub.file_download._download_to_tmp_and_move",
            side_effect=AssertionError("Unexpected network download"),
        ),
    ):
        result = hf_hub_download(
            repo_id="test/repo",
            filename="model.bin",
            local_dir=local_dir,
            force_download=False,
        )

    assert Path(result) == destination
    assert destination.read_bytes() == new_content
    assert cached_file.read_bytes() == new_content
    assert not list(local_dir.glob(".model.bin.*"))


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="POSIX file permissions are required",
)
def test_cache_copy_preserves_normal_file_permissions(tmp_path: Path):
    import stat

    from huggingface_hub.file_download import _chmod_and_move

    destination = tmp_path / "model.bin"
    destination.write_bytes(b"old content")
    destination.chmod(0o644)

    temporary = tmp_path / ".model.bin.tmp"
    temporary.touch(mode=0o600)
    temporary.write_bytes(b"new content")

    _chmod_and_move(temporary, destination)

    reference = tmp_path / "reference.bin"
    reference.touch()

    assert destination.read_bytes() == b"new content"
    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


def test_cache_copy_falls_back_when_replace_fails(tmp_path: Path):
    import os

    from huggingface_hub.file_download import _replace_no_matter_what

    destination = tmp_path / "model.bin"
    destination.write_bytes(b"old content")

    temporary = tmp_path / ".model.bin.tmp"
    temporary.write_bytes(b"new content")

    real_replace = os.replace

    def fail_replace_over_existing(src, dst):
        if Path(dst) == destination and destination.exists():
            raise OSError("Replace-over-existing is not supported")
        return real_replace(src, dst)

    with patch(
        "huggingface_hub.file_download.os.replace",
        side_effect=fail_replace_over_existing,
    ):
        _replace_no_matter_what(temporary, destination)

    assert destination.read_bytes() == b"new content"
    assert not temporary.exists()
    assert not list(tmp_path.glob(".model.bin.*.old"))
    assert not list(tmp_path.glob(".model.bin.*.new"))
