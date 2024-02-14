from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ._base import FILENAME_PATTERN, MAX_SHARD_SIZE, split_state_dict_into_shards


if TYPE_CHECKING:
    import numpy as np


def split_numpy_state_dict_into_shards(
    state_dict: Dict[str, "np.ndarray"],
    *,
    filename_pattern: str = FILENAME_PATTERN,
    max_shard_size: int = MAX_SHARD_SIZE,
) -> Tuple[Dict[str, Dict[str, "np.ndarray"]], Optional[Dict]]:
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
        state_dict (`Dict[str, np.ndarray]`):
            The state dictionary to save.
        max_shard_size (`int` or `str`, *optional*):
            The maximum size of each shard, in bytes. Defaults to 5GB.
        filename_pattern (`str`, *optional*):
            The pattern to generate the files names in which the model will be saved. Pattern must be a string that
            can be formatted with `filename_pattern.format(suffix=...)` and must contain the keyword `suffix`
            Defaults to `"model{suffix}.safetensors"`.
    """
    return split_state_dict_into_shards(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern=filename_pattern,
        get_tensor_size=get_tensor_size,
    )


def get_tensor_size(tensor: "np.ndarray") -> int:
    return tensor.nbytes
