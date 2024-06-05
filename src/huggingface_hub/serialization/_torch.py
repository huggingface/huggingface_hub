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
"""Contains pytorch-specific helpers."""

import importlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from .. import constants, logging
from ._base import MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory


logger = logging.get_logger(__file__)

if TYPE_CHECKING:
    import torch


def split_torch_state_dict_into_shards(
    state_dict: Dict[str, "torch.Tensor"],
    *,
    filename_pattern: str = constants.SAFETENSORS_WEIGHTS_FILE_PATTERN,
    max_shard_size: Union[int, str] = MAX_SHARD_SIZE,
) -> StateDictSplit:
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
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.

    Returns:
        [`StateDictSplit`]: A `StateDictSplit` object containing the shards and the index to retrieve them.

    Example:
    ```py
    >>> import json
    >>> import os
    >>> from safetensors.torch import save_file as safe_save_file
    >>> from huggingface_hub import split_torch_state_dict_into_shards

    >>> def save_state_dict(state_dict: Dict[str, torch.Tensor], save_directory: str):
    ...     state_dict_split = split_torch_state_dict_into_shards(state_dict)
    ...     for filename, tensors in state_dict_split.filename_to_tensors.items():
    ...         shard = {tensor: state_dict[tensor] for tensor in tensors}
    ...         safe_save_file(
    ...             shard,
    ...             os.path.join(save_directory, filename),
    ...             metadata={"format": "pt"},
    ...         )
    ...     if state_dict_split.is_sharded:
    ...         index = {
    ...             "metadata": state_dict_split.metadata,
    ...             "weight_map": state_dict_split.tensor_to_filename,
    ...         }
    ...         with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
    ...             f.write(json.dumps(index, indent=2))
    ```
    """
    return split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern=filename_pattern,
        get_tensor_size=get_tensor_size,
        get_storage_id=get_torch_storage_id,
    )


def save_torch_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    save_directory: Union[str, Path],
    *,
    safe_serialization: bool = True,
    filename_pattern: Optional[str] = None,
    max_shard_size: Union[int, str] = MAX_SHARD_SIZE,
) -> None:
    save_directory = str(save_directory)

    # Imports correct library
    if safe_serialization:
        filename_pattern = constants.SAFETENSORS_WEIGHTS_FILE_PATTERN

        try:
            from safetensors.torch import save_file as save_file_fn
        except ImportError as e:
            raise ImportError(
                "Please install `safetensors` to use safe serialization. "
                "You can install it with `pip install safetensors`."
            ) from e

    else:
        filename_pattern = constants.PYTORCH_WEIGHTS_FILE_PATTERN

        from torch import save as save_file_fn  # type: ignore[assignment]

        logger.warning(
            "You are using unsafe serialization. Due to security reasons, it is recommended not to load "
            "pickled models from untrusted sources. If you intend to share your model, we strongly recommend "
            "using safe serialization by installing `safetensors` with `pip install safetensors`."
        )

    # Split dict
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )

    # Clean the folder from previous save
    existing_files_regex = re.compile(filename_pattern.format(suffix=r"(-\d{5}-of-\d{5})?") + r"(\.index\.json)?")
    for filename in os.listdir(save_directory):
        if existing_files_regex.match(filename):
            try:
                logger.debug(f"Removing existing file '{filename}' from folder.")
                os.remove(os.path.join(save_directory, filename))
            except Exception as e:
                logger.warning(f"Error when trying to remove existing '{filename}' from folder: {e}. Continuing...")

    # Save each shard
    safe_file_kwargs = {"metadata": {"format": "pt"}} if safe_serialization else {}
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        save_file_fn(shard, os.path.join(save_directory, filename), **safe_file_kwargs)
        logger.debug(f"Shard saved to {filename}")

    # Save the index (if any)
    if state_dict_split.is_sharded:
        index_path = filename_pattern.format(suffix="") + ".index.json"
        index = {"metadata": state_dict_split.metadata, "weight_map": state_dict_split.tensor_to_filename}
        with open(os.path.join(save_directory, index_path), "w") as f:
            json.dump(index, f, indent=2)
        logger.info(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}). "
            f"Model weighs have been saved in {len(state_dict_split.filename_to_tensors)} checkpoint shards. "
            f"You can find where each parameters has been saved in the index located at {index_path}."
        )

    logger.info(f"Model weights successfully saved to {save_directory}!")


def get_torch_storage_id(tensor: "torch.Tensor") -> Tuple["torch.device", int, int]:
    """
    Return unique identifier to a tensor storage.

    Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    Taken from https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/pytorch_utils.py#L278.
    """
    if tensor.device.type == "xla" and is_torch_tpu_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, get_storage_size(tensor)


def get_tensor_size(tensor: "torch.Tensor") -> int:
    return tensor.numel() * tensor.element_size()


@lru_cache()
def is_torch_tpu_available(check_device=True):
    """
    Checks if `torch_xla` is installed and potentially if a TPU is in the environment

    Taken from https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/utils/import_utils.py#L463.
    """
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # We need to check if `xla_device` can be found, will raise a RuntimeError if not
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


def storage_ptr(tensor: "torch.Tensor") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L11C1-L20C21.
    """
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def get_storage_size(tensor: "torch.Tensor") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L31C1-L41C59
    """
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _get_dtype_size(tensor.dtype)
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _get_dtype_size(tensor.dtype)


@lru_cache()
def _get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L344
    """
    import torch

    # torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]
