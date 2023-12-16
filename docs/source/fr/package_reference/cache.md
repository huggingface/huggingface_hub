<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Référence au système cache

Le système de caching a été mis à jour dans la version v0.8.0 pour devenir un système
cache central et partagé par toutes les librairies dépendant du Hub. Consultez le
[guide système cache](../guides/manage-cache) pour une présentation détaillée du
cache à HF.

## Les Helpers

### try_to_load_from_cache

[[autodoc]] huggingface_hub.try_to_load_from_cache

### cached_assets_path

[[autodoc]] huggingface_hub.cached_assets_path

### scan_cache_dir

[[autodoc]] huggingface_hub.scan_cache_dir

## Structures de données

Toutes les structures sont construites et retournées par [`scan_cache_dir`]
et sont immuables.

### HFCacheInfo

[[autodoc]] huggingface_hub.HFCacheInfo

### CachedRepoInfo

[[autodoc]] huggingface_hub.CachedRepoInfo
    - size_on_disk_str
    - refs

### CachedRevisionInfo

[[autodoc]] huggingface_hub.CachedRevisionInfo
    - size_on_disk_str
    - nb_files

### CachedFileInfo

[[autodoc]] huggingface_hub.CachedFileInfo
    - size_on_disk_str

### DeleteCacheStrategy

[[autodoc]] huggingface_hub.DeleteCacheStrategy
    - expected_freed_size_str

## Exceptions

### CorruptedCacheException

[[autodoc]] huggingface_hub.CorruptedCacheException