# tests/test_upload_large_folder.py
import pytest

from huggingface_hub._upload_large_folder import COMMIT_SIZE_SCALE, LargeUploadStatus


@pytest.fixture
def status():
    return LargeUploadStatus(items=[])


def test_target_chunk_default(status):
    assert status.target_chunk() == COMMIT_SIZE_SCALE[1]


@pytest.mark.parametrize(
    "start_idx, success, delta_items, duration, expected_idx",
    [
        (2, False, 0, 10, 1),  # drop by one on failure
        (0, False, 0, 10, 0),  # never go below zero
        (1, True, 0, 50, 1),  # duration >= 40 --> no bump
        (1, True, -1, 30, 1),  # nb_items < threshold --> no bump
        (1, True, 0, 30, 2),  # fast enough and enough items
        (len(COMMIT_SIZE_SCALE) - 1, True, 0, 10, len(COMMIT_SIZE_SCALE) - 1),  # never exceed last index
    ],
)
def test_update_chunk_transitions(status, start_idx, success, delta_items, duration, expected_idx):
    status._chunk_idx = start_idx
    threshold = COMMIT_SIZE_SCALE[start_idx]
    nb_items = threshold + delta_items
    status.update_chunk(success=success, nb_items=nb_items, duration=duration)

    assert status._chunk_idx == expected_idx
    assert status.target_chunk() == COMMIT_SIZE_SCALE[expected_idx]
