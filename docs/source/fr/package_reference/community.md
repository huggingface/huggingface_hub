<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interagir avec les dicussions et les pull requestions

Regardez la page de documentation [`HfApi`] pour les références des méthodes permettant l'intéraction
avec des pull requests et des discussions sur le Hub.

- [`get_repo_discussions`]
- [`get_discussion_details`]
- [`create_discussion`]
- [`create_pull_request`]
- [`rename_discussion`]
- [`comment_discussion`]
- [`edit_discussion_comment`]
- [`change_discussion_status`]
- [`merge_pull_request`]

## Structure des données

[[autodoc]] Discussion

[[autodoc]] DiscussionWithDetails

[[autodoc]] DiscussionEvent

[[autodoc]] DiscussionComment

[[autodoc]] DiscussionStatusChange

[[autodoc]] DiscussionCommit

[[autodoc]] DiscussionTitleChange
