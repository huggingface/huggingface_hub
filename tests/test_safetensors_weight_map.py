from huggingface_hub.errors import SafetensorsParsingError
from huggingface_hub.utils._safetensors import (
    SafetensorsFileMetadata,
    TensorInfo,
    _assert_weight_map_matches_shard_headers,
)


def _file_meta(*names: str) -> SafetensorsFileMetadata:
    tensors = {
        name: TensorInfo(dtype="F32", shape=[2, 2], data_offsets=(0, 16)) for name in names
    }
    return SafetensorsFileMetadata(metadata={"format": "pt"}, tensors=tensors)


def test_weight_map_matching_headers_is_ok() -> None:
    shard = "model-00001-of-00002.safetensors"
    _assert_weight_map_matches_shard_headers(
        {"w.weight": shard, "w.bias": shard},
        {shard: _file_meta("w.weight", "w.bias")},
    )


def test_weight_map_missing_tensor_raises() -> None:
    shard = "model-00001-of-00002.safetensors"
    try:
        _assert_weight_map_matches_shard_headers(
            {"w.weight": shard, "mtp.fc.weight": shard},
            {shard: _file_meta("w.weight")},
        )
    except SafetensorsParsingError as exc:
        assert "mtp.fc.weight" in str(exc)
        assert shard in str(exc)
    else:
        raise AssertionError("expected SafetensorsParsingError")
