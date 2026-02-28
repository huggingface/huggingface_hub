<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# API Filesystem

La classe `HfFileSystem` fournit une interface fichier pythonique vers le Hugging Face Hub basée sur [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/).

## HfFileSystem

`HfFileSystem` est basé sur [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), il est donc compatible avec la plupart des APIs qu'il offre. Pour plus de détails, consultez [notre guide](../guides/hf_file_system) et la [référence API](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem) de fsspec.

[[autodoc]] HfFileSystem 
    - __init__
    - all
