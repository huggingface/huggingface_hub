"""
Helper classes to modify pipeline outputs from tensors to expected pipeline output
"""

from typing import TYPE_CHECKING, Dict, List, Union


Classes = Dict[str, Union[str, float]]

if TYPE_CHECKING:
    try:
        import torch
    except Exception:
        pass


def speaker_diarization_normalize(
    tensor: "torch.Tensor", sampling_rate: int, classnames: List[str]
) -> List[Classes]:
    N = tensor.shape[1]
    if len(classnames) != N:
        raise ValueError(
            f"There is a mismatch between classnames ({len(classnames)}) and number of speakers ({N})"
        )
    classes = []
    for i in range(N):
        values, counts = tensor[:, i].unique_consecutive(return_counts=True)
        offset = 0
        for v, c in zip(values, counts):
            if v == 1:
                classes.append(
                    {
                        "class": classnames[i],
                        "start": offset / sampling_rate,
                        "end": (offset + c.item()) / sampling_rate,
                    }
                )
            offset += c.item()

    classes = sorted(classes, key=lambda x: x["start"])
    return classes
