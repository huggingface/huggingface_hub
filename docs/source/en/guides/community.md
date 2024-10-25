<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interact with Discussions and Pull Requests

The `huggingface_hub` library provides a Python interface to interact with Pull Requests and Discussions on the Hub.
Visit [the dedicated documentation page](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
for a deeper view of what Discussions and Pull Requests on the Hub are, and how they work under the hood.

## Retrieve Discussions and Pull Requests from the Hub

The `HfApi` class allows you to retrieve Discussions and Pull Requests on a given repo:

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

`HfApi.get_repo_discussions` supports filtering by author, type (Pull Request or Discussion) and status (`open` or `closed`):

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

`HfApi.get_repo_discussions` returns a [generator](https://docs.python.org/3.7/howto/functional.html#generators) that yields
[`Discussion`] objects. To get all the Discussions in a single list, run:

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

The [`Discussion`] object returned by [`HfApi.get_repo_discussions`] contains high-level overview of the
Discussion or Pull Request. You can also get more detailed information using [`HfApi.get_discussion_details`]:

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

[`HfApi.get_discussion_details`] returns a [`DiscussionWithDetails`] object, which is a subclass of [`Discussion`]
with more detailed information about the Discussion or Pull Request. Information includes all the comments, status changes,
and renames of the Discussion via [`DiscussionWithDetails.events`].

In case of a Pull Request, you can retrieve the raw git diff with [`DiscussionWithDetails.diff`]. All the commits of the
Pull Requests are listed in [`DiscussionWithDetails.events`].


## Create and edit a Discussion or Pull Request programmatically

The [`HfApi`] class also offers ways to create and edit Discussions and Pull Requests.
You will need an [access token](https://huggingface.co/docs/hub/security-tokens) to create and edit Discussions
or Pull Requests.

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
