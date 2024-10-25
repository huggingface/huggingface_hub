<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 파일 다운로드 하기[[downloading-files]]

## 단일 파일 다운로드하기[[download-a-single-file]]

### hf_hub_download[[huggingface_hub.hf_hub_download]]

[[autodoc]]huggingface_hub.hf_hub_download

### hf_hub_url[[huggingface_hub.hf_hub_url]]

[[autodoc]]huggingface_hub.hf_hub_url

## 리포지토리의 스냅샷 다운로드하기[[huggingface_hub.snapshot_download]]

[[autodoc]]huggingface_hub.snapshot_download

## 파일에 대한 메타데이터 가져오기[[get-metadata-about-a-file]]

### get_hf_file_metadata[[huggingface_hub.get_hf_file_metadata]]

[[autodoc]]huggingface_hub.get_hf_file_metadata

### HfFileMetadata[[huggingface_hub.HfFileMetadata]]

[[autodoc]]huggingface_hub.HfFileMetadata

## 캐싱[[caching]]

위에 나열된 메소드들은 파일을 재다운로드하지 않도록 하는 캐싱 시스템과 함께 작동하도록 설계되었습니다. v0.8.0에서의 업데이트로, 캐싱 시스템은 Hub를 기반으로 하는 다양한 라이브러리 간의 공유 중앙 캐시 시스템으로 발전했습니다.

Hugging Face에서의 캐싱에 대한 자세한 설명은[캐시 시스템 가이드](../guides/manage-cache)를 참조하세요.
