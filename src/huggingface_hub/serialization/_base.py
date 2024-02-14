# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains helpers to split tensors into shards."""
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar


TensorT = TypeVar("TensorT")
TensorSizeFn_T = Callable[[TensorT], int]
StorageIDFn_T = Callable[[TensorT], Optional[Any]]

MAX_SHARD_SIZE = 5_000_000
FILENAME_PATTERN = "model{suffix}.safetensors"


def split_state_dict_into_shards(
    state_dict: Dict[str, TensorT],
    *,
    get_tensor_size: TensorSizeFn_T,
    filename_pattern: str = FILENAME_PATTERN,
    max_shard_size: int = MAX_SHARD_SIZE,
    get_storage_id: StorageIDFn_T = lambda tensor: None,
) -> Tuple[Dict[str, Dict[str, TensorT]], Optional[Dict]]:
    """
    Split a model state dictionary in shards so that each shard is smaller than a given size.

    The shards are determined by iterating through the `state_dict` in the order of its keys. There is no optimization
    made to make each shard as close as possible to the maximum size passed. For example, if the limit is 10GB and we
    have tensors of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB], [6+2+2GB] and not
    [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's tensor is bigger than `max_shard_size`, it will end up in its own shard which will have a
    size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, Tensor]`):
            The state dictionary to save.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
        get_tensor_size (`Callable[[Tensor], int]`):
            A function that returns the size of a tensor in bytes.
        get_storage_id (`Callable[[Tensor], Optional[Any]]`, *optional*):
            A function that returns a unique identifier to a tensor storage. Multiple different tensors can share the
            same underlying storage. This identifier is guaranteed to be unique and constant for this tensor's storage
            during its lifetime. Two tensor storages with non-overlapping lifetimes may have the same id.
    """
    storage_id_to_shard_id: Dict[Any, int] = {}

    shard_list: List[Dict[str, TensorT]] = []
    current_shard: Dict[str, TensorT] = {}
    current_shard_size = 0
    total_size = 0

    for key, tensor in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(tensor, str):
            continue

        # If a `tensor` shares the same underlying storage as another tensor, we put `tensor` in the same `block`
        storage_id = get_storage_id(tensor)
        if storage_id is not None and storage_id in storage_id_to_shard_id:
            shard_id = storage_id_to_shard_id[storage_id]
            shard_list[shard_id][key] = tensor
            continue

        # Compute tensor size
        tensor_size = get_tensor_size(tensor)

        # If this tensor is bigger than the maximal size, we put it in its own shard
        if tensor_size > max_shard_size:
            shard_list.append({key: tensor})
            continue

        # If this tensor is going to tip up over the maximal size, we split.
        # Current shard already has some tensors, we add it to the list of shards and create a new one.
        if current_shard_size + tensor_size > max_shard_size:
            shard_list.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        # Add the tensor to the current shard
        current_shard[key] = tensor
        current_shard_size += tensor_size
        total_size += tensor_size
        storage_id_to_shard_id[storage_id] = len(shard_list)

    # Add the last shard
    if len(current_shard) > 0:
        shard_list.append(current_shard)
    nb_shards = len(shard_list)

    # If we only have one shard, we return it => no need to build the index
    if nb_shards == 1:
        filename = filename_pattern.format(suffix="")
        return ({filename: shard_list[0]}, None)

    # Now that each tensor is assigned to a shard, let's assign a filename to each shard
    tensor_name_to_filename = {}
    filename_to_tensors = {}
    for idx, shard in enumerate(shard_list):
        filename = filename_pattern.format(suffix=f"-{idx+1:05d}-of-{nb_shards:05d}")
        for key in shard:
            tensor_name_to_filename[key] = filename
        filename_to_tensors[filename] = shard

    # Build the index and return
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": tensor_name_to_filename,
    }
    return (filename_to_tensors, index)


# ##############################
# # Framework-specific helpers #
# ##############################


# def get_storage_id(tensor: TensorT) -> Any:
#     return _get_framework_helpers(tensor).get_storage_id(tensor)


# def get_tensor_size(tensor: TensorT) -> int:
#     return _get_framework_helpers(tensor).get_tensor_size(tensor)


# def _get_framework_helpers(tensor: TensorT):
#     """
#     Returns framework-specific helpers depending on the tensor type.

#     Supports PyTorch, TensorFlow and Numpy.
#     """
#     if is_torch_available():
#         import torch

#         if isinstance(tensor, torch.Tensor):
#             from . import _pytorch_helpers

#             return _pytorch_helpers

#     if is_tf_available():
#         import tensorflow as tf

#         if isinstance(tensor, tf.Tensor):
#             from . import _tensorflow_helpers

#             return _tensorflow_helpers

#     if is_numpy_available():
#         import numpy as np

#         if isinstance(tensor, np.ndarray):
#             from . import _numpy_helpers

#             return _numpy_helpers

#     raise ValueError(f"Unknown tensor type. Only Torch, TensorFlow and Numpy are supported, not '{type(tensor)}'.")
