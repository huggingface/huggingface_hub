<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Managing local and online repositories

La classe `Repository` est un helper autour des commandes `git` et `git-lfs`. Elle offre des outils adaptés
à la gestion de dépôts qui peuvent être très volumineux. 

C'est l'outil recommandé dès que des opérations avec `git` sont faites, ou lorsque la collaboration sera un point
clef du dépôt.

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

## Commandes asynchrones

L'utilitaire `Repository` offre plusieurs méthodes qui peuvent tourner en asynchrone:
- `git_push`
- `git_pull`
- `push_to_hub`
- Le manager de contexte `commit`

Regardez ci-dessous les utilities pour gérer ce genre de méthodes asynchrones.

[[autodoc]] Repository
    - commands_failed
    - commands_in_progress
    - wait_for_commands

[[autodoc]] huggingface_hub.repository.CommandInProgress