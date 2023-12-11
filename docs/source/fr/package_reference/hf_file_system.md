<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# L'API FileSystem

La classe `HfFileSystem` offre une interface Python pour le Hub Hugging Face basée sur [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/).

## HfFileSystem

`HfFileSystem` est basé sur [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), donc cette classe est compatible avec la plupart des API offertes par fsspec. Pour plus de détails, consultez [notre guide](../guides/hf_file_system) et les [Références](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem) de l'API fsspec.

[[autodoc]] HfFileSystem
    - __init__
    - resolve_path
    - ls
