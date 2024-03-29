<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 파일 다운로드 중[[downloading-files]]

## 단일 파일 다운로드[[download-a-single-file]]

### hf_hub_다운로드[[huggingface_hub.hf_hub_download]]

[[autodoc]]huggingface_hub.hf_hub_download

### hf_hub_url[[huggingface_hub.hf_hub_url]]

[[autodoc]]huggingface_hub.hf_hub_url

## 저장소의 스냅샷 다운로드[[huggingface_hub.snapshot_download]]

[[autodoc]]huggingface_hub.snapshot_download

## 파일에 대한 메타데이터 가져오기[[get-metadata-about-a-file]]

### get_hf_file_metadata[[huggingface_hub.get_hf_file_metadata]]

[[autodoc]]huggingface_hub.get_hf_file_metadata

### HfFileMetadata[[huggingface_hub.HfFileMetadata]]

[[autodoc]]huggingface_hub.HfFileMetadata

## 캐싱[[caching]]

위에 표시된 방법은 다음을 방지하는 캐싱 시스템과 함께 작동하도록 설계되었습니다.
파일을 다시 다운로드하는 중입니다. 캐싱 시스템은 v0.8.0에서 업데이트되어 중앙 시스템이 되었습니다.
허브에 의존하는 라이브러리 전체에서 공유되는 캐시 시스템입니다.

캐싱에 대한 자세한 내용은 [캐시 시스템 가이드](../guides/manage-cache)를 읽어보세요.
HF에서.
