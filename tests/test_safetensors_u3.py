# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from typing import get_args

from huggingface_hub.utils._safetensors import DTYPE_T, SafetensorsFileMetadata, TensorInfo


def test_u3_parameter_count_is_read_from_logical_shape() -> None:
    assert "U3" in get_args(DTYPE_T)

    elements = 9
    physical_bytes = (3 * elements + 7) // 8
    metadata = SafetensorsFileMetadata(
        metadata={"format": "pt"},
        tensors={
            "packed.weight": TensorInfo(
                dtype="U3",
                shape=[elements],
                data_offsets=(0, physical_bytes),
            )
        },
    )

    assert metadata.parameter_count == {"U3": elements}
