import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from huggingface_hub import constants
from huggingface_hub.serialization import (
    get_tf_storage_size,
    get_torch_storage_size,
    save_torch_model,
    save_torch_state_dict,
    split_state_dict_into_shards_factory,
    split_torch_state_dict_into_shards,
)
from huggingface_hub.serialization._base import parse_size_to_int

from .testing_utils import requires


if TYPE_CHECKING:
    import torch


def _dummy_get_storage_id(item):
    return None


def _dummy_get_storage_size(item):
    return sum(item)


# util functions for checking the version for pytorch
def is_wrapper_tensor_subclass_available():
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def dummy_state_dict() -> Dict[str, List[int]]:
    return {
        "layer_1": [6],
        "layer_2": [10],
        "layer_3": [30],
        "layer_4": [2],
        "layer_5": [2],
    }


@pytest.fixture
def torch_state_dict() -> Dict[str, "torch.Tensor"]:
    try:
        import torch

        return {
            "layer_1": torch.tensor([4]),
            "layer_2": torch.tensor([10]),
            "layer_3": torch.tensor([30]),
            "layer_4": torch.tensor([2]),
            "layer_5": torch.tensor([2]),
        }
    except ImportError:
        pytest.skip("torch is not available")


@pytest.fixture
def torch_state_dict_tensor_subclass() -> Dict[str, "torch.Tensor"]:
    try:
        import torch
        from torch.testing._internal.two_tensor import TwoTensor

        t = torch.tensor([4])
        return {
            "layer_1": torch.tensor([4]),
            "layer_2": torch.tensor([10]),
            "layer_3": torch.tensor([30]),
            "layer_4": torch.tensor([2]),
            "layer_5": torch.tensor([2]),
            "layer_6": TwoTensor(t, t),
        }
    except ImportError:
        pytest.skip("torch is not available")


@pytest.fixture
def torch_state_dict_shared_layers() -> Dict[str, "torch.Tensor"]:
    try:
        import torch

        shared_layer = torch.tensor([4])

        return {
            "shared_1": shared_layer,
            "unique_1": torch.tensor([10]),
            "unique_2": torch.tensor([30]),
            "shared_2": shared_layer,
        }
    except ImportError:
        pytest.skip("torch is not available")


@pytest.fixture
def torch_state_dict_shared_layers_tensor_subclass() -> Dict[str, "torch.Tensor"]:
    try:
        import torch
        from torch.testing._internal.two_tensor import TwoTensor

        t = torch.tensor([4])
        tensor_subclass_tensor = TwoTensor(t, t)

        t = torch.tensor([4])
        shared_tensor_subclass_tensor = TwoTensor(t, t)
        return {
            "layer_1": torch.tensor([4]),
            "layer_2": torch.tensor([10]),
            "layer_3": torch.tensor([30]),
            "layer_4": torch.tensor([2]),
            "layer_5": torch.tensor([2]),
            "layer_6": tensor_subclass_tensor,
            "ts_shared_1": shared_tensor_subclass_tensor,
            "ts_shared_2": shared_tensor_subclass_tensor,
        }
    except ImportError:
        pytest.skip("torch is not available")


def test_single_shard(dummy_state_dict):
    state_dict_split = split_state_dict_into_shards_factory(
        dummy_state_dict,
        get_storage_id=_dummy_get_storage_id,
        get_storage_size=_dummy_get_storage_size,
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


def test_multiple_shards(dummy_state_dict):
    state_dict_split = split_state_dict_into_shards_factory(
        dummy_state_dict,
        get_storage_id=_dummy_get_storage_id,
        get_storage_size=_dummy_get_storage_size,
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
        get_storage_size=_dummy_get_storage_size,
        max_shard_size=1,
        filename_pattern="model{suffix}.safetensors",
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


@requires("tensorflow")
def test_get_tf_storage_size():
    import tensorflow as tf

    assert get_tf_storage_size(tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)) == 5 * 8
    assert get_tf_storage_size(tf.constant([1, 2, 3, 4, 5], dtype=tf.float16)) == 5 * 2


@requires("torch")
def test_get_torch_storage_size():
    import torch

    assert get_torch_storage_size(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)) == 5 * 8
    assert get_torch_storage_size(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float16)) == 5 * 2


@requires("torch")
@pytest.mark.skipif(not is_wrapper_tensor_subclass_available(), reason="requires torch 2.1 or higher")
def test_get_torch_storage_size_wrapper_tensor_subclass():
    import torch
    from torch.testing._internal.two_tensor import TwoTensor

    t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
    assert get_torch_storage_size(TwoTensor(t, t)) == 5 * 8 * 2
    t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float16)
    assert get_torch_storage_size(TwoTensor(t, TwoTensor(t, t))) == 5 * 2 * 3


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


def test_save_torch_model(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test `save_torch_model` is only a wrapper around `save_torch_state_dict`."""
    model_mock = Mock()
    safe_state_dict_mock = mocker.patch("huggingface_hub.serialization._torch.save_torch_state_dict")
    save_torch_model(
        model_mock,
        save_directory=tmp_path,
        filename_pattern="my-pattern",
        force_contiguous=True,
        max_shard_size="3GB",
        metadata={"foo": "bar"},
        safe_serialization=True,
    )
    safe_state_dict_mock.assert_called_once_with(
        state_dict=model_mock.state_dict.return_value,
        save_directory=tmp_path,
        filename_pattern="my-pattern",
        force_contiguous=True,
        max_shard_size="3GB",
        metadata={"foo": "bar"},
        safe_serialization=True,
    )


def test_save_torch_state_dict_not_sharded(tmp_path: Path, torch_state_dict: Dict[str, "torch.Tensor"]) -> None:
    """Save as safetensors without sharding."""
    save_torch_state_dict(torch_state_dict, tmp_path, max_shard_size="1GB")
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "model.safetensors.index.json").is_file()


def test_save_torch_state_dict_sharded(tmp_path: Path, torch_state_dict: Dict[str, "torch.Tensor"]) -> None:
    """Save as safetensors with sharding."""
    save_torch_state_dict(torch_state_dict, tmp_path, max_shard_size=30)
    assert not (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "model.safetensors.index.json").is_file()
    assert (tmp_path / "model-00001-of-00002.safetensors").is_file()
    assert (tmp_path / "model-00001-of-00002.safetensors").is_file()

    assert json.loads((tmp_path / "model.safetensors.index.json").read_text("utf-8")) == {
        "metadata": {"total_size": 40},
        "weight_map": {
            "layer_1": "model-00001-of-00002.safetensors",
            "layer_2": "model-00001-of-00002.safetensors",
            "layer_3": "model-00001-of-00002.safetensors",
            "layer_4": "model-00002-of-00002.safetensors",
            "layer_5": "model-00002-of-00002.safetensors",
        },
    }


def test_save_torch_state_dict_unsafe_not_sharded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, torch_state_dict: Dict[str, "torch.Tensor"]
) -> None:
    """Save as pickle without sharding."""
    with caplog.at_level("WARNING"):
        save_torch_state_dict(torch_state_dict, tmp_path, max_shard_size="1GB", safe_serialization=False)
    assert "we strongly recommend using safe serialization" in caplog.text

    assert (tmp_path / "pytorch_model.bin").is_file()
    assert not (tmp_path / "pytorch_model.bin.index.json").is_file()


@pytest.mark.skipif(not is_wrapper_tensor_subclass_available(), reason="requires torch 2.1 or higher")
def test_save_torch_state_dict_tensor_subclass_unsafe_not_sharded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, torch_state_dict_tensor_subclass: Dict[str, "torch.Tensor"]
) -> None:
    """Save as pickle without sharding."""
    with caplog.at_level("WARNING"):
        save_torch_state_dict(
            torch_state_dict_tensor_subclass, tmp_path, max_shard_size="1GB", safe_serialization=False
        )
    assert "we strongly recommend using safe serialization" in caplog.text

    assert (tmp_path / "pytorch_model.bin").is_file()
    assert not (tmp_path / "pytorch_model.bin.index.json").is_file()


@pytest.mark.skipif(not is_wrapper_tensor_subclass_available(), reason="requires torch 2.1 or higher")
def test_save_torch_state_dict_shared_layers_tensor_subclass_unsafe_not_sharded(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    torch_state_dict_shared_layers_tensor_subclass: Dict[str, "torch.Tensor"],
) -> None:
    """Save as pickle without sharding."""
    with caplog.at_level("WARNING"):
        save_torch_state_dict(
            torch_state_dict_shared_layers_tensor_subclass, tmp_path, max_shard_size="1GB", safe_serialization=False
        )
    assert "we strongly recommend using safe serialization" in caplog.text

    assert (tmp_path / "pytorch_model.bin").is_file()
    assert not (tmp_path / "pytorch_model.bin.index.json").is_file()


def test_save_torch_state_dict_unsafe_sharded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, torch_state_dict: Dict[str, "torch.Tensor"]
) -> None:
    """Save as pickle with sharding."""
    # Check logs
    with caplog.at_level("WARNING"):
        save_torch_state_dict(torch_state_dict, tmp_path, max_shard_size=30, safe_serialization=False)
    assert "we strongly recommend using safe serialization" in caplog.text

    assert not (tmp_path / "pytorch_model.bin").is_file()
    assert (tmp_path / "pytorch_model.bin.index.json").is_file()
    assert (tmp_path / "pytorch_model-00001-of-00002.bin").is_file()
    assert (tmp_path / "pytorch_model-00001-of-00002.bin").is_file()

    assert json.loads((tmp_path / "pytorch_model.bin.index.json").read_text("utf-8")) == {
        "metadata": {"total_size": 40},
        "weight_map": {
            "layer_1": "pytorch_model-00001-of-00002.bin",
            "layer_2": "pytorch_model-00001-of-00002.bin",
            "layer_3": "pytorch_model-00001-of-00002.bin",
            "layer_4": "pytorch_model-00002-of-00002.bin",
            "layer_5": "pytorch_model-00002-of-00002.bin",
        },
    }


def test_save_torch_state_dict_shared_layers_not_sharded(
    tmp_path: Path, torch_state_dict_shared_layers: Dict[str, "torch.Tensor"]
) -> None:
    from safetensors.torch import load_file

    save_torch_state_dict(torch_state_dict_shared_layers, tmp_path, safe_serialization=True)
    safetensors_file = tmp_path / "model.safetensors"
    assert safetensors_file.is_file()

    # Check shared layer not duplicated in file
    state_dict = load_file(safetensors_file)
    assert "shared_1" in state_dict
    assert "shared_2" not in state_dict

    # Check shared layer info in metadata
    file_bytes = safetensors_file.read_bytes()
    metadata_str = file_bytes[
        8 : struct.unpack("<Q", file_bytes[:8])[0] + 8
    ].decode()  # TODO: next time add helper for this
    assert json.loads(metadata_str)["__metadata__"]["shared_2"] == "shared_1"


def test_save_torch_state_dict_shared_layers_sharded(
    tmp_path: Path, torch_state_dict_shared_layers: Dict[str, "torch.Tensor"]
) -> None:
    from safetensors.torch import load_file

    save_torch_state_dict(torch_state_dict_shared_layers, tmp_path, max_shard_size=2, safe_serialization=True)
    index_file = tmp_path / "model.safetensors.index.json"
    assert index_file.is_file()

    # Check shared layer info in index metadata
    index = json.loads(index_file.read_text())
    assert index["metadata"]["shared_2"] == "shared_1"

    # Check shared layer not duplicated in files
    for filename in index["weight_map"].values():
        state_dict = load_file(tmp_path / filename)
        assert "shared_2" not in state_dict


def test_split_torch_state_dict_into_shards(
    tmp_path: Path, torch_state_dict_shared_layers_tensor_subclass: Dict[str, "torch.Tensor"]
):
    # the model size is 72, setting max_shard_size to 32 means we'll shard the file
    state_dict_split = split_torch_state_dict_into_shards(
        torch_state_dict_shared_layers_tensor_subclass,
        filename_pattern=constants.PYTORCH_WEIGHTS_FILE_PATTERN,
        max_shard_size=32,
    )
    assert state_dict_split.is_sharded


def test_save_torch_state_dict_custom_filename(tmp_path: Path, torch_state_dict: Dict[str, "torch.Tensor"]) -> None:
    """Custom filename pattern is respected."""
    # Not sharded
    save_torch_state_dict(torch_state_dict, tmp_path, filename_pattern="model.variant{suffix}.safetensors")
    assert (tmp_path / "model.variant.safetensors").is_file()

    # Sharded
    save_torch_state_dict(
        torch_state_dict, tmp_path, filename_pattern="model.variant{suffix}.safetensors", max_shard_size=30
    )
    assert (tmp_path / "model.variant.safetensors.index.json").is_file()
    assert (tmp_path / "model.variant-00001-of-00002.safetensors").is_file()
    assert (tmp_path / "model.variant-00002-of-00002.safetensors").is_file()


def test_save_torch_state_dict_delete_existing_files(
    tmp_path: Path, torch_state_dict: Dict[str, "torch.Tensor"]
) -> None:
    """Directory is cleaned before saving new files."""
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").touch()
    (tmp_path / "model-00001-of-00003.safetensors").touch()
    (tmp_path / "model-00002-of-00003.safetensors").touch()
    (tmp_path / "model-00003-of-00003.safetensors").touch()

    (tmp_path / "pytorch_model.bin").touch()
    (tmp_path / "pytorch_model.bin.index.json").touch()
    (tmp_path / "pytorch_model-00001-of-00003.bin").touch()
    (tmp_path / "pytorch_model-00002-of-00003.bin").touch()
    (tmp_path / "pytorch_model-00003-of-00003.bin").touch()

    save_torch_state_dict(torch_state_dict, tmp_path)
    assert (tmp_path / "model.safetensors").stat().st_size > 0  # new file

    # Previous shards have been deleted
    assert not (tmp_path / "model.safetensors.index.json").is_file()  # deleted
    assert not (tmp_path / "model-00001-of-00003.safetensors").is_file()  # deleted
    assert not (tmp_path / "model-00002-of-00003.safetensors").is_file()  # deleted
    assert not (tmp_path / "model-00003-of-00003.safetensors").is_file()  # deleted

    # But not previous pickle files (since saving as safetensors)
    assert (tmp_path / "pytorch_model.bin").is_file()  # not deleted
    assert (tmp_path / "pytorch_model.bin.index.json").is_file()
    assert (tmp_path / "pytorch_model-00001-of-00003.bin").is_file()
    assert (tmp_path / "pytorch_model-00002-of-00003.bin").is_file()
    assert (tmp_path / "pytorch_model-00003-of-00003.bin").is_file()
