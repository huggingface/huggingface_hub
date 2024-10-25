<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Discussions 및 Pull Requests를 이용하여 상호작용하기[[interact-with-discussions-and-pull-requests]]

`huggingface_hub` 라이브러리는 Hub의 Pull Requests 및 Discussions와 상호작용할 수 있는 Python 인터페이스를 제공합니다.
[전용 문서 페이지](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)를 방문하여 Hub의 Discussions와 Pull Requests가 무엇이고 어떻게 작동하는지 자세히 살펴보세요.

## Hub에서 Discussions 및 Pull Requests 가져오기[[retrieve-discussions-and-pull-requests-from-the-hub]]

`HfApi` 클래스를 사용하면 지정된 리포지토리에 대한 Discussions 및 Pull Requests를 검색할 수 있습니다:

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(repo_id="bigscience/bloom"):
...     print(f"{discussion.num} - {discussion.title}, pr: {discussion.is_pull_request}")

# 11 - Add Flax weights, pr: True
# 10 - Update README.md, pr: True
# 9 - Training languages in the model card, pr: True
# 8 - Update tokenizer_config.json, pr: True
# 7 - Slurm training script, pr: False
[...]
```

`HfApi.get_repo_discussion`은 작성자, 유형(Pull Requests 또는 Discussion) 및 상태(`open` 또는 `closed`)별로 필터링을 지원합니다:

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(
...    repo_id="bigscience/bloom",
...    author="ArthurZ",
...    discussion_type="pull_request",
...    discussion_status="open",
... ):
...     print(f"{discussion.num} - {discussion.title} by {discussion.author}, pr: {discussion.is_pull_request}")

# 19 - Add Flax weights by ArthurZ, pr: True
```

`HfApi.get_repo_discussions`는 [`Discussion`] 객체를 생성하는 [생성자](https://docs.python.org/3.7/howto/functional.html#generators)를 반환합니다. 모든 Discussions를 하나의 리스트로 가져오려면 다음을 실행합니다:

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

[`HfApi.get_repo_discussions`]가 반환하는 [`Discussion`] 객체에는 Discussions 또는 Pull Request에 대한 개략적인 개요가 포함되어 있습니다. [`HfApi.get_discussion_details`]를 사용하여 더 자세한 정보를 얻을 수도 있습니다:

```python
>>> from huggingface_hub import get_discussion_details

>>> get_discussion_details(
...     repo_id="bigscience/bloom-1b3",
...     discussion_num=2
... )
DiscussionWithDetails(
    num=2,
    author='cakiki',
    title='Update VRAM memory for the V100s',
    status='open',
    is_pull_request=True,
    events=[
        DiscussionComment(type='comment', author='cakiki', ...),
        DiscussionCommit(type='commit', author='cakiki', summary='Update VRAM memory for the V100s', oid='1256f9d9a33fa8887e1c1bf0e09b4713da96773a', ...),
    ],
    conflicting_files=[],
    target_branch='refs/heads/main',
    merge_commit_oid=None,
    diff='diff --git a/README.md b/README.md\nindex a6ae3b9294edf8d0eda0d67c7780a10241242a7e..3a1814f212bc3f0d3cc8f74bdbd316de4ae7b9e3 100644\n--- a/README.md\n+++ b/README.md\n@@ -132,7 +132,7 [...]',
)
```

[`HfApi.get_discussion_details`]는 Discussion 또는 Pull Request에 대한 자세한 정보가 포함된 [`Discussion`]의 하위 클래스인 [`DiscussionWithDetails`] 객체를 반환합니다. 해당 정보는 [`DiscussionWithDetails.events`]를 통해 Discussion의 모든 댓글, 상태 변경 및 이름 변경을 포함하고 있습니다.

Pull Request의 경우, [`DiscussionWithDetails.diff`]를 통해 원시 git diff를 검색할 수 있습니다. Pull Request의 모든 커밋은 [`DiscussionWithDetails.events`]에 나열됩니다.


## 프로그래밍 방식으로 Discussion 또는 Pull Request를 생성하고 수정하기[[create-and-edit-a-discussion-or-pull-request-programmatically]]

[`HfApi`] 클래스는 Discussions 및 Pull Requests를 생성하고 수정하는 방법도 제공합니다.
Discussions와 Pull Requests를 만들고 편집하려면 [접근 토큰](https://huggingface.co/docs/hub/security-tokens)이 필요합니다.

Hub의 리포지토리에 변경 사항을 제안하는 가장 간단한 방법은 [`create_commit`] API를 사용하는 것입니다. `create_pr` 매개변수를 `True`로 설정하기만 하면 됩니다. 이 매개변수는 [`create_commit`]을 래핑하는 다른 함수에서도 사용할 수 있습니다:

    * [`upload_file`]
    * [`upload_folder`]
    * [`delete_file`]
    * [`delete_folder`]
    * [`metadata_update`]

```python
>>> from huggingface_hub import metadata_update

>>> metadata_update(
...     repo_id="username/repo_name",
...     metadata={"tags": ["computer-vision", "awesome-model"]},
...     create_pr=True,
... )
```

리포지토리에 대한 Discussion(또는 Pull Request)을 만들려면 [`HfApi.create_discussion`](또는 [`HfApi.create_pull_request`])을 사용할 수도 있습니다.
이 방법으로 Pull Request를 열면 로컬에서 변경 작업을 해야 하는 경우에 유용할 수 있습니다. 이 방법으로 열린 Pull Request는 `"draft"` 모드가 됩니다.

```python
>>> from huggingface_hub import create_discussion, create_pull_request

>>> create_discussion(
...     repo_id="username/repo-name",
...     title="Hi from the huggingface_hub library!",
...     token="<insert your access token here>",
... )
DiscussionWithDetails(...)

>>> create_pull_request(
...     repo_id="username/repo-name",
...     title="Hi from the huggingface_hub library!",
...     token="<insert your access token here>",
... )
DiscussionWithDetails(..., is_pull_request=True)
```

Pull Requests 및 Discussions 관리는 전적으로 [`HfApi`] 클래스로 할 수 있습니다. 예를 들어:

    * 댓글을 추가하려면 [`comment_discussion`]
    * 댓글을 수정하려면 [`edit_discussion_comment`]
    * Discussion 또는 Pull Request의 이름을 바꾸려면 [`rename_discussion`]
    * Discussion / Pull Request를 열거나 닫으려면 [`change_discussion_status`]
    * Pull Request를 병합하려면 [`merge_pull_request`]를 사용합니다.


사용 가능한 모든 메소드에 대한 전체 참조는 [`HfApi`] 문서 페이지를 참조하세요.

## Pull Request에 변경 사항 푸시[[push-changes-to-a-pull-request]]

*곧 공개됩니다!*

## 참고 항목[[see-also]]

더 자세한 내용은 [Discussions 및 Pull Requests](../package_reference/community)와 [hf_api](../package_reference/hf_api) 문서 페이지를 참조하세요.
