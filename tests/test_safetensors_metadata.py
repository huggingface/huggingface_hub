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
import json
import re

import pytest

from huggingface_hub import HfApi, constants
from huggingface_hub.utils import SafetensorsFileMetadata, SafetensorsParsingError, TensorInfo


def _file_metadata(*tensor_names: str) -> SafetensorsFileMetadata:
    return SafetensorsFileMetadata(
        metadata={"format": "pt"},
        tensors={
            name: TensorInfo(dtype="F32", shape=[1], data_offsets=(index * 4, (index + 1) * 4))
            for index, name in enumerate(tensor_names)
        },
    )


def test_get_safetensors_metadata_rejects_tensor_missing_from_mapped_shard(tmp_path, mocker) -> None:
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    index_path = tmp_path / constants.SAFETENSORS_INDEX_FILE
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "first.weight": first_shard,
                    "misplaced.weight": first_shard,
                    "second.weight": second_shard,
                }
            }
        )
    )

    api = HfApi()
    mocker.patch.object(
        api,
        "file_exists",
        side_effect=lambda *, filename, **kwargs: filename == constants.SAFETENSORS_INDEX_FILE,
    )
    mocker.patch.object(api, "hf_hub_download", return_value=str(index_path))
    shard_metadata = {
        first_shard: _file_metadata("first.weight"),
        # Keep the missing tensor in another shard to prove that the index's exact mapping is validated.
        second_shard: _file_metadata("misplaced.weight", "second.weight"),
    }
    mocker.patch.object(
        api,
        "parse_safetensors_file_metadata",
        side_effect=lambda *, filename, **kwargs: shard_metadata[filename],
    )

    with pytest.raises(
        SafetensorsParsingError,
        match=re.escape(
            "Safetensors index for 'namespace/repo' is inconsistent: tensor 'misplaced.weight' is mapped to "
            f"'{first_shard}' but is not present in that file."
        ),
    ):
        api.get_safetensors_metadata("namespace/repo")
