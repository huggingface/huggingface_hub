<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Managing local and online repositories

The `Repository` class is a helper class that wraps `git` and `git-lfs` commands. It provides tooling adapted
for managing repositories which can be very large.

It is the recommended tool as soon as any `git` operation is involved, or when collaboration will be a point
of focus with the repository itself.

## The Repository class

[[autodoc]] Repository
    - __init__
    - current_branch
    - all

## Helper methods

[[autodoc]] huggingface_hub.repository.is_git_repo

[[autodoc]] huggingface_hub.repository.is_local_clone

[[autodoc]] huggingface_hub.repository.is_tracked_with_lfs

[[autodoc]] huggingface_hub.repository.is_git_ignored

[[autodoc]] huggingface_hub.repository.files_to_be_staged

[[autodoc]] huggingface_hub.repository.is_tracked_upstream

[[autodoc]] huggingface_hub.repository.commits_to_push

## Following asynchronous commands

The `Repository` utility offers several methods which can be launched asynchronously:
- `git_push`
- `git_pull`
- `push_to_hub`
- The `commit` context manager

See below for utilities to manage such asynchronous methods.

[[autodoc]] Repository
    - commands_failed
    - commands_in_progress
    - wait_for_commands

[[autodoc]] huggingface_hub.repository.CommandInProgress
