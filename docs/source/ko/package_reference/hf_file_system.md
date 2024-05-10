<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 파일 시스템 API[[filesystem-api]]

[`HfFileSystem`] 클래스는 [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/)을 기반으로 Hugging Face Hub에 Python 파일 인터페이스를 제공합니다.

## [HfFileSystem](Hf파일시스템)

[`HfFileSystem`]은 [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/)을 기반으로 하므로 제공되는 대부분의 API와 호환됩니다. 자세한 내용은 [가이드](../guides/hf_file_system) 및 fsspec의 [API 레퍼런스](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem)를 확인하세요.

[[autodoc]] HfFileSystem
    - __init__
    - resolve_path
    - ls
