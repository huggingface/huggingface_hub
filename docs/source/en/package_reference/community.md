<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interacting with Discussions and Pull Requests

Check the [`HfApi`] documentation page for the reference of methods enabling
interaction with Pull Requests and Discussions on the Hub.

- [`get_repo_discussions`]
- [`get_discussion_details`]
- [`create_discussion`]
- [`create_pull_request`]
- [`rename_discussion`]
- [`comment_discussion`]
- [`edit_discussion_comment`]
- [`change_discussion_status`]
- [`merge_pull_request`]

## Data structures

[[autodoc]] Discussion

[[autodoc]] DiscussionWithDetails

[[autodoc]] DiscussionEvent

[[autodoc]] DiscussionComment

[[autodoc]] DiscussionStatusChange

[[autodoc]] DiscussionCommit

[[autodoc]] DiscussionTitleChange
