<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Filesystem API

The `HfFileSystem` class provides a pythonic file interface to the Hugging Face Hub based on [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/).

## HfFileSystem

`HfFileSystem` is based on [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), so it is compatible with most of the APIs that it offers. For more details, check out [our guide](../guides/hf_file_system) and fsspec's [API Reference](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem).

[[autodoc]] HfFileSystem

## HfFileSystemEditFile

In addition to regular file-like objects obtained using open modes "w", "wb", "r" or "rb" to read and overwrite files, `HfFileSystem` also offers open modes "a" and "ab" to append to an existing file and "e" and "eb" to edit an existing file in-place.

[[autodoc]] huggingface_hub.hf_file_system.HfFileSystemEditFile
