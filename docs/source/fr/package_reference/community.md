<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interagir avec les Discussions et Pull Requests

Consultez la page de documentation [`HfApi`] pour la référence des méthodes permettant
d'interagir avec les Pull Requests et Discussions sur le Hub.

- [`get_repo_discussions`]
- [`get_discussion_details`]
- [`create_discussion`]
- [`create_pull_request`]
- [`rename_discussion`]
- [`comment_discussion`]
- [`edit_discussion_comment`]
- [`change_discussion_status`]
- [`merge_pull_request`]

## Structures de données

[[autodoc]] Discussion

[[autodoc]] DiscussionWithDetails

[[autodoc]] DiscussionEvent

[[autodoc]] DiscussionComment

[[autodoc]] DiscussionStatusChange

[[autodoc]] DiscussionCommit

[[autodoc]] DiscussionTitleChange
