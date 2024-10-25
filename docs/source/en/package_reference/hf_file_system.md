<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Filesystem API

The `HfFileSystem` class provides a pythonic file interface to the Hugging Face Hub based on [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/).

## HfFileSystem

`HfFileSystem` is based on [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), so it is compatible with most of the APIs that it offers. For more details, check out [our guide](../guides/hf_file_system) and fsspec's [API Reference](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem).

[[autodoc]] HfFileSystem 
    - __init__
    - all
