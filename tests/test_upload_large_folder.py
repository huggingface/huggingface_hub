# tests/test_upload_large_folder.py
import pytest

from huggingface_hub._upload_large_folder import COMMIT_SIZE_SCALE, LargeUploadStatus


@pytest.fixture
def status():
    return LargeUploadStatus(items=[])


def test_target_chunk_default(status):
    assert status.target_chunk() == COMMIT_SIZE_SCALE[1]


def test_decrease_chunk_on_failure(status):
    status._chunk_idx = 2
    status.update_chunk(success=False, nb_items=0, duration=10)
    assert status._chunk_idx == 1


def test_decrease_chunk_not_below_zero(status):
    status._chunk_idx = 0
    status.update_chunk(success=False, nb_items=0, duration=10)
    assert status._chunk_idx == 0


def test_increase_chunk_on_success_and_fast(status):
    idx = 1
    status._chunk_idx = idx
    threshold = COMMIT_SIZE_SCALE[idx]
    status.update_chunk(success=True, nb_items=threshold, duration=30)
    assert status._chunk_idx == idx + 1


def test_no_increase_chunk_on_slow(status):
    idx = 1
    status._chunk_idx = idx
    threshold = COMMIT_SIZE_SCALE[idx]
    status.update_chunk(success=True, nb_items=threshold, duration=50)
    assert status._chunk_idx == idx


def test_no_increase_if_insufficient_items(status):
    idx = 1
    status._chunk_idx = idx
    threshold = COMMIT_SIZE_SCALE[idx]
    status.update_chunk(success=True, nb_items=threshold - 1, duration=30)
    assert status._chunk_idx == idx


def test_not_above_max(status):
    last = len(COMMIT_SIZE_SCALE) - 1
    status._chunk_idx = last
    threshold = COMMIT_SIZE_SCALE[last]
    status.update_chunk(success=True, nb_items=threshold, duration=10)
    assert status._chunk_idx == last


def test_target_chunk_after_update(status):
    last = len(COMMIT_SIZE_SCALE) - 1
    status._chunk_idx = last - 1
    threshold = COMMIT_SIZE_SCALE[last - 1]
    status.update_chunk(success=True, nb_items=threshold, duration=10)
    assert status.target_chunk() == COMMIT_SIZE_SCALE[last]
