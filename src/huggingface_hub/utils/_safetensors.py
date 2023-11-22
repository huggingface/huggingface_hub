from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


FILENAME_T = str
TENSOR_NAME_T = str
DTYPE_T = Literal["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"]


@dataclass
class TensorInfo:
    dtype: DTYPE_T
    shape: List[int]
    data_offsets: Tuple[int, int]


@dataclass
class SafetensorsFileMetadata:
    metadata: Dict
    tensors: Dict[TENSOR_NAME_T, TensorInfo]

    @classmethod
    def from_raw(cls, data: Dict) -> "SafetensorsFileMetadata":
        return cls(
            metadata=data["__metadata__"],
            tensors={
                key: TensorInfo(
                    dtype=tensor["dtype"],
                    shape=tensor["shape"],
                    data_offsets=tuple(tensor["data_offsets"]),  # type: ignore
                )
                for key, tensor in data.items()
                if key != "__metadata__"
            },
        )


@dataclass
class SafetensorsRepoMetadata:
    metadata: Optional[Dict]
    sharded: bool
    weight_map: Dict[TENSOR_NAME_T, FILENAME_T]  # tensor name -> filename
    files_metadata: Dict[FILENAME_T, SafetensorsFileMetadata]  # filename -> metadata
