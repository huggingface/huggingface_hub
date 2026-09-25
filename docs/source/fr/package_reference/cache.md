<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Référence du système de cache

Le système de mise en cache a été mis à jour dans v0.8.0 pour devenir le système de cache central partagé
entre les bibliothèques qui dépendent du Hub. Lisez le [guide du système de cache](../guides/manage-cache)
pour une présentation détaillée de la mise en cache chez HF.

## Helpers

### try_to_load_from_cache

[[autodoc]] huggingface_hub.try_to_load_from_cache

### cached_assets_path

[[autodoc]] huggingface_hub.cached_assets_path

### scan_cache_dir

[[autodoc]] huggingface_hub.scan_cache_dir

## Structures de données

Toutes les structures sont construites et retournées par [`scan_cache_dir`] et sont immuables.

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
