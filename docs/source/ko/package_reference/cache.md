<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 캐시 시스템 참조[[cache-system-reference]]

버전 0.8.0에서의 업데이트로, 캐시 시스템은 Hub에 의존하는 라이브러리 전체에서 공유되는 중앙 캐시 시스템으로 발전하였습니다. Hugging Face 캐싱에 대한 자세한 설명은 [캐시 시스템 가이드](../guides/manage-cache)를 참조하세요.

## 도우미 함수[[helpers]]

### try_to_load_from_cache[[huggingface_hub.try_to_load_from_cache]]

[[autodoc]] huggingface_hub.try_to_load_from_cache

### cached_assets_path[[huggingface_hub.cached_assets_path]]

[[autodoc]] huggingface_hub.cached_assets_path

### scan_cache_dir[[huggingface_hub.scan_cache_dir]]

[[autodoc]] huggingface_hub.scan_cache_dir

## 데이터 구조[[data-structures]]

모든 구조체는 [`scan_cache_dir`]에 의해 생성되고 반환되며, 불변(immutable)입니다.

### HFCacheInfo[[huggingface_hub.HFCacheInfo]]

[[autodoc]] huggingface_hub.HFCacheInfo

### CachedRepoInfo[[huggingface_hub.CachedRepoInfo]]

[[autodoc]] huggingface_hub.CachedRepoInfo
    - size_on_disk_str
    - refs

### CachedRevisionInfo[[huggingface_hub.CachedRevisionInfo]]

[[autodoc]] huggingface_hub.CachedRevisionInfo
    - size_on_disk_str
    - nb_files

### CachedFileInfo[[huggingface_hub.CachedFileInfo]]

[[autodoc]] huggingface_hub.CachedFileInfo
    - size_on_disk_str

### DeleteCacheStrategy[[huggingface_hub.DeleteCacheStrategy]]

[[autodoc]] huggingface_hub.DeleteCacheStrategy
    - expected_freed_size_str

## 예외[[exceptions]]

### CorruptedCacheException[[huggingface_hub.CorruptedCacheException]]

[[autodoc]] huggingface_hub.CorruptedCacheException
