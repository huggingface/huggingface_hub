<!--⚠️ 请注意，该文件是 Markdown 格式，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确渲染。
-->

# 互动讨论与拉取请求（Pull Request）

huggingface_hub 库提供了一个 Python 接口，用于与 Hub 上的拉取请求（Pull Request）和讨论互动。
访问 [相关的文档页面](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
，了解有关 Hub 上讨论和拉取请求（Pull Request）的更深入的介绍及其工作原理。

## 从 Hub 获取讨论和拉取请求（Pull Request）

`HfApi` 类允许您获取给定仓库中的讨论和拉取请求（Pull Request）：

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

`HfApi.get_repo_discussions` 支持按作者、类型（拉取请求或讨论）和状态（`open` 或 `closed`）进行过滤：

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

`HfApi.get_repo_discussions` 返回一个 [生成器](https://docs.python.org/3.7/howto/functional.html#generators) 生成
[`Discussion`] 对象。 要获取所有讨论并存储为列表，可以运行：

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

[`HfApi.get_repo_discussions`] 返回的 [`Discussion`] 对象提供讨论或拉取请求（Pull Request）的高级概览。您还可以使用 [`HfApi.get_discussion_details`] 获取更详细的信息：

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

[`HfApi.get_discussion_details`] 返回一个 [`DiscussionWithDetails`] 对象，它是 [`Discussion`] 的子类，包含有关讨论或拉取请求（Pull Request）的更详细信息。详细信息包括所有评论、状态更改以及讨论的重命名信息，可通过 [`DiscussionWithDetails.events`] 获取。

如果是拉取请求（Pull Request），您可以通过 [`DiscussionWithDetails.diff`] 获取原始的 git diff。拉取请求（Pull Request）的所有提交都列在 [`DiscussionWithDetails.events`] 中。


## 以编程方式创建和编辑讨论或拉取请求

[`HfApi`] 类还提供了创建和编辑讨论及拉取请求（Pull Request）的方法。
您需要一个 [访问令牌](https://huggingface.co/docs/hub/security-tokens) 来创建和编辑讨论或拉取请求（Pull Request）。

在 Hub 上对 repo 提出修改建议的最简单方法是使用 [`create_commit`] API：只需将 `create_pr` 参数设置为 `True` 。此参数也适用于其他封装了 [`create_commit`] 的方法：

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

您还可以使用 [`HfApi.create_discussion`]（或 [`HfApi.create_pull_request`]）在仓库上创建讨论（或拉取请求）。以这种方式打开拉取请求在您需要本地处理更改时很有用。以这种方式打开的拉取请求将处于“draft”模式。

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

您可以使用 [`HfApi`] 类来方便地管理拉取请求和讨论。例如：

    * [`comment_discussion`] 添加评论
    * [`edit_discussion_comment`] 编辑评论
    * [`rename_discussion`] 重命名讨论或拉取请求
    * [`change_discussion_status`] 打开或关闭讨论/拉取请求
    * [`merge_pull_request`] 合并拉取请求


请访问 [`HfApi`] 文档页面，获取所有可用方法的完整参考

## 推送更改到拉取请求（Pull Request）

*敬请期待！*

## 参见

有关更详细的参考，请访问 [讨论和拉取请求](../package_reference/community) 和 [hf_api](../package_reference/hf_api) 文档页面。
