import re
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from .utils._runtime import is_numpy_available, is_tf_available, is_torch_available


TensorT = TypeVar("TensorT")

#############################
# Framework-agnostic helper #
#############################


def split_state_dict_into_shards(
    state_dict: Dict[str, TensorT],
    max_shard_size: int = 5_000_000,
    # Examples for filename_pattern:
    # - "model{suffix}.safetensors"
    # - "pytorch_model{suffix}.bin"
    # - "tf_model{suffix}.h5"
    filename_pattern: str = "model{suffix}.safetensors",
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
        state_dict (`Dict[str, torch.Tensor]`):
            The state dictionary to save.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
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
        storage_id = _get_storage_id(tensor)
        if storage_id is not None and storage_id in storage_id_to_shard_id:
            shard_id = storage_id_to_shard_id[storage_id]
            shard_list[shard_id][key] = tensor
            continue

        # Compute tensor size
        tensor_size = _get_tensor_size(tensor)

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


##############################
# Framework-specific helpers #
##############################


def _get_storage_id(tensor: TensorT) -> Any:
    """
    Returns the storage id of `tensor` if it has one, else `None`.

    Storage id is used to determine if two tensors share the same underlying storage.
    When that's the case, we put them in the same shard.
    """
    if is_torch_available():
        # TODO: implement this for torch (see https://github.com/huggingface/transformers/blob/74d9d0cebb0263a3f8ab9c280569170cc74651d0/src/transformers/pytorch_utils.py#L283)
        pass
    return None


def _get_tensor_size(tensor: TensorT) -> int:
    """
    Returns the size (in bytes) occupied by `tensor`.

    Supports PyTorch, TensorFlow and Numpy tensors.
    """
    if is_torch_available():
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.numel() * tensor.element_size()

    if is_tf_available():
        import tensorflow as tf

        if isinstance(tensor, tf.Tensor):
            return tensor.numpy().size * _dtype_byte_size_tf(tensor.dtype)

    if is_numpy_available():
        import numpy as np

        if isinstance(tensor, np.ndarray):
            return tensor.nbytes

    raise ValueError(f"Unknown tensor type. Only Torch, TensorFlow and Numpy are supported, not {type(tensor)}")


def _dtype_byte_size_tf(dtype) -> float:
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Taken from https://github.com/huggingface/transformers/blob/74d9d0cebb0263a3f8ab9c280569170cc74651d0/src/transformers/modeling_tf_utils.py#L608.
    NOTE: why not `tensor.numpy().nbytes`?

    Example:

    ```py
    >>> dtype_byte_size(tf.float32)
    4
    ```
    """
    import tensorflow as tf

    if dtype == tf.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", dtype.name)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8
