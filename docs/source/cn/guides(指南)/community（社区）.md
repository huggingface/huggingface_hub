<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interact with Discussions and Pull Requests   与讨论和拉取请求进行交互

The `huggingface_hub` library provides a Python interface to interact with Pull Requests and Discussions on the Hub.
Visit [the dedicated documentation page](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
for a deeper view of what Discussions and Pull Requests on the Hub are, and how they work under the hood.

huggingface_hub 库提供了一个 Python 接口来与 Hub 上的拉取请求和讨论进行交互。请访问专门的文档页面,以更深入地了解 Hub 上的讨论和拉取请求是什么以及它们的工作原理。

## Retrieve Discussions and Pull Requests from the Hub  从Hub检索讨论和拉取请求

The `HfApi` class allows you to retrieve Discussions and Pull Requests on a given repo:

Hf Api允许您检索给定仓库上的讨论和拉取请求:

```python
>>> from huggingface_hub import get_repo_discussions  #从 Hugging Face Hub 库中导入 get_repo_discussions 函数。
>>> for discussion in get_repo_discussions   (repo_id="bigscience/bloom-1b3"):
...     print(f"{discussion.num} - {discussion.title}, pr: {discussion.is_pull_request}")  #调用该函数，获取存储库 ID 为 "bigscience/bloom-1b3" 的讨论列表，使用 for 循环迭代讨论列表中的每个讨论，discussion.num: 讨论的编号。discussion.title: 讨论的标题。discussion.is_pull_request: 一个布尔值，指示讨论是否与拉取请求（pull request）相关。

# 11 - Add Flax weights, pr: True
# 10 - Update README.md, pr: True
# 9 - Training languages in the model card, pr: True
# 8 - Update tokenizer_config.json, pr: True
# 7 - Slurm training script, pr: False
[...]
```

`HfApi.get_repo_discussions` returns a [generator](https://docs.python.org/3.7/howto/functional.html#generators) that yields
[`Discussion`] objects. To get all the Discussions in a single list, run:

HfApi.get_repo_discussions 返回一个生成器，它会生成 [Discussion] 对象。要将所有讨论获取为一个列表，请运行：

```python
>>> from huggingface_hub import get_repo_discussions  #从 Hugging Face Hub 库中导入 get_repo_discussions 函数。
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))  #调用该函数，获取存储库 ID 为 "bert-base-uncased" 的讨论生成器，将生成器转换为列表，通过迭代获取所有讨论对象，并将它们存储在 discussions_list 变量中。
```

The [`Discussion`] object returned by [`HfApi.get_repo_discussions`] contains high-level overview of the
Discussion or Pull Request. You can also get more detailed information using [`HfApi.get_discussion_details`]:


[Discussion] 对象由 [HfApi.get_repo_discussions] 返回，包含讨论或拉取请求的高级概览。您还可以使用 [HfApi.get_discussion_details] 获取更详细的信息。

```python
>>> from huggingface_hub import get_discussion_details #从 Hugging Face Hub 库中导入 get_discussion_details 函数
>>> get_discussion_details(
...     repo_id="bigscience/bloom-1b3",
...     discussion_num=2
... ) #指定存储库 ID 为 "bigscience/bloom-1b3",指定要获取详细信息的讨论的编号为 2,
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
)  #返回一个包含详细信息的 DiscussionWithDetails 对象，其中包含以下属性：num=2: 讨论的编号。author='cakiki': 讨论的作者。title='Update VRAM memory for the V100s': 讨论的标题。status='open': 讨论的状态（开放）。is_pull_request=True: 表示这是一个拉取请求。events=[...]: 讨论的事件列表，包括评论和提交等。conflicting_files=[]: 冲突的文件列表。target_branch='refs/heads/main': 目标分支。merge_commit_oid=None: 合并提交的对象标识符。diff='...': 差异信息，显示文件的更改。
```

[`HfApi.get_discussion_details`] returns a [`DiscussionWithDetails`] object, which is a subclass of [`Discussion`]
with more detailed information about the Discussion or Pull Request. Information includes all the comments, status changes,
and renames of the Discussion via [`DiscussionWithDetails.events`].

[HfApi.get_discussion_details] 返回一个 [DiscussionWithDetails] 对象，它是 [Discussion] 的子类，提供有关讨论或拉取请求的更详细信息。这些信息包括讨论的所有评论、状态更改以及通过 [DiscussionWithDetails.events] 进行的重命名。

In case of a Pull Request, you can retrieve the raw git diff with [`DiscussionWithDetails.diff`]. All the commits of the
Pull Request are listed in [`DiscussionWithDetails.events`].


在拉取请求的情况下，您可以使用 [DiscussionWithDetails.diff] 获取原始的 Git 差异。拉取请求的所有提交都列在[DiscussionWithDetails.events] 中。

## Create and edit a Discussion or Pull Request programmatically   通过编程方式创建和编辑讨论或拉取请求

The [`HfApi`] class also offers ways to create and edit Discussions and Pull Requests.
You will need an [access token](https://huggingface.co/docs/hub/security-tokens) to create and edit Discussions
or Pull Requests.

[HfApi] 类还提供了创建和编辑讨论以及拉取请求的方法。您将需要一个访问令牌来创建和编辑讨论或拉取请求。

The simplest way to propose changes on a repo on the Hub is via the [`create_commit`] API: just 
set the `create_pr` parameter to `True`. This parameter is also available on other methods that wrap [`create_commit`]:

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

You can also use [`HfApi.create_discussion`] (respectively [`HfApi.create_pull_request`]) to create a Discussion (respectively a Pull Request) on a repo.
Opening a Pull Request this way can be useful if you need to work on changes locally. Pull Requests opened this way will be in `"draft"` mode.

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

Managing Pull Requests and Discussions can be done entirely with the [`HfApi`] class. For example:

    * [`comment_discussion`] to add comments
    * [`edit_discussion_comment`] to edit comments
    * [`rename_discussion`] to rename a Discussion or Pull Request 
    * [`change_discussion_status`] to open or close a Discussion / Pull Request 
    * [`merge_pull_request`] to merge a Pull Request 


Visit the [`HfApi`] documentation page for an exhaustive reference of all available methods.

## Push changes to a Pull Request

*Coming soon !*

## See also

For a more detailed reference, visit the [Discussions and Pull Requests](../package_reference/community) and the [hf_api](../package_reference/hf_api) documentation page.
