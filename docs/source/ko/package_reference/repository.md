<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 로컬 및 온라인 저장소 관리[[managing-local-and-online-repositories]]

`Repository` 클래스는 `git` 및 `git-lfs` 명령을 래핑하는 도우미 클래스입니다. 적합한 툴링을 제공합니다.
매우 큰 저장소를 관리하는 데 사용됩니다.

`git` 작업이 포함되거나 협업이 중요한 경우 즉시 권장되는 도구입니다.
저장소 자체에 중점을 둡니다.

## 리포지토리 클래스[[the-repository-class]]

[[autodoc]] Repository
    - __init__
    - current_branch
    - all

## 도우미 방법[[helper-methods]]

[[autodoc]] huggingface_hub.repository.is_git_repo

[[autodoc]] huggingface_hub.repository.is_local_clone

[[autodoc]] huggingface_hub.repository.is_tracked_with_lfs

[[autodoc]] huggingface_hub.repository.is_git_ignored

[[autodoc]] huggingface_hub.repository.files_to_be_staged

[[autodoc]] huggingface_hub.repository.is_tracked_upstream

[[autodoc]] huggingface_hub.repository.commits_to_push

## 비동기 명령 따르기[[following-asynchronous-commands]]

`Repository` 유틸리티는 비동기적으로 시작할 수 있는 여러 메서드를 제공합니다.
- `git_push`
- `git_pull`
- `push_to_hub`
- `commit` 컨텍스트 관리자

이러한 비동기 메서드를 관리하는 유틸리티는 아래를 참조하세요.

[[autodoc]] Repository
    - commands_failed
    - commands_in_progress
    - wait_for_commands

[[autodoc]] huggingface_hub.repository.CommandInProgress
