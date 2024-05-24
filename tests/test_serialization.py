import pytest

from huggingface_hub.serialization import split_state_dict_into_shards_factory
from huggingface_hub.serialization._base import parse_size_to_int
from huggingface_hub.serialization._numpy import get_tensor_size as get_tensor_size_numpy
from huggingface_hub.serialization._tensorflow import get_tensor_size as get_tensor_size_tensorflow
from huggingface_hub.serialization._torch import get_tensor_size as get_tensor_size_torch

from .testing_utils import requires


DUMMY_STATE_DICT = {
    "layer_1": [6],
    "layer_2": [10],
    "layer_3": [30],
    "layer_4": [2],
    "layer_5": [2],
}


def _dummy_get_storage_id(item):
    return None


def _dummy_get_tensor_size(item):
    return sum(item)


def test_single_shard():
    state_dict_split = split_state_dict_into_shards_factory(
        DUMMY_STATE_DICT,
        get_storage_id=_dummy_get_storage_id,
        get_tensor_size=_dummy_get_tensor_size,
        max_shard_size=100,  # large shard size => only one shard
        filename_pattern="file{suffix}.dummy",
    )
    assert not state_dict_split.is_sharded
    assert state_dict_split.filename_to_tensors == {
        # All layers fit in one shard => no suffix in filename
        "file.dummy": ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"],
    }
    assert state_dict_split.tensor_to_filename == {
        "layer_1": "file.dummy",
        "layer_2": "file.dummy",
        "layer_3": "file.dummy",
        "layer_4": "file.dummy",
        "layer_5": "file.dummy",
    }
    assert state_dict_split.metadata == {"total_size": 50}


def test_multiple_shards():
    state_dict_split = split_state_dict_into_shards_factory(
        DUMMY_STATE_DICT,
        get_storage_id=_dummy_get_storage_id,
        get_tensor_size=_dummy_get_tensor_size,
        max_shard_size=10,  # small shard size => multiple shards
        filename_pattern="file{suffix}.dummy",
    )

    assert state_dict_split.is_sharded
    assert state_dict_split.filename_to_tensors == {
        # layer 4 and 5 could go in this one but assignment is not optimal, and it's fine
        "file-00001-of-00004.dummy": ["layer_1"],
        "file-00002-of-00004.dummy": ["layer_3"],
        "file-00003-of-00004.dummy": ["layer_2"],
        "file-00004-of-00004.dummy": ["layer_4", "layer_5"],
    }
    assert state_dict_split.tensor_to_filename == {
        "layer_1": "file-00001-of-00004.dummy",
        "layer_3": "file-00002-of-00004.dummy",
        "layer_2": "file-00003-of-00004.dummy",
        "layer_4": "file-00004-of-00004.dummy",
        "layer_5": "file-00004-of-00004.dummy",
    }
    assert state_dict_split.metadata == {"total_size": 50}


def test_tensor_same_storage():
    state_dict_split = split_state_dict_into_shards_factory(
        {
            "layer_1": [1],
            "layer_2": [2],
            "layer_3": [1],
            "layer_4": [2],
            "layer_5": [1],
        },
        get_storage_id=lambda x: (x[0]),  # dummy for test: storage id based on first element
        get_tensor_size=_dummy_get_tensor_size,
        max_shard_size=1,
    )
    assert state_dict_split.is_sharded
    assert state_dict_split.filename_to_tensors == {
        "model-00001-of-00002.safetensors": ["layer_2", "layer_4"],
        "model-00002-of-00002.safetensors": ["layer_1", "layer_3", "layer_5"],
    }
    assert state_dict_split.tensor_to_filename == {
        "layer_1": "model-00002-of-00002.safetensors",
        "layer_2": "model-00001-of-00002.safetensors",
        "layer_3": "model-00002-of-00002.safetensors",
        "layer_4": "model-00001-of-00002.safetensors",
        "layer_5": "model-00002-of-00002.safetensors",
    }
    assert state_dict_split.metadata == {"total_size": 3}  # count them once


@requires("numpy")
def test_get_tensor_size_numpy():
    import numpy as np

    assert get_tensor_size_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64)) == 5 * 8
    assert get_tensor_size_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float16)) == 5 * 2


@requires("tensorflow")
def test_get_tensor_size_tensorflow():
    import tensorflow as tf

    assert get_tensor_size_tensorflow(tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)) == 5 * 8
    assert get_tensor_size_tensorflow(tf.constant([1, 2, 3, 4, 5], dtype=tf.float16)) == 5 * 2


@requires("torch")
def test_get_tensor_size_torch():
    import torch

    assert get_tensor_size_torch(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)) == 5 * 8
    assert get_tensor_size_torch(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float16)) == 5 * 2


def test_parse_size_to_int():
    assert parse_size_to_int("1KB") == 1 * 10**3
    assert parse_size_to_int("2MB") == 2 * 10**6
    assert parse_size_to_int("3GB") == 3 * 10**9
    assert parse_size_to_int(" 10 KB ") == 10 * 10**3  # ok with whitespace
    assert parse_size_to_int("20mb") == 20 * 10**6  # ok with lowercase

    with pytest.raises(ValueError, match="Unit 'IB' not supported"):
        parse_size_to_int("1KiB")  # not a valid unit

    with pytest.raises(ValueError, match="Could not parse the size value"):
        parse_size_to_int("1ooKB")  # not a float
