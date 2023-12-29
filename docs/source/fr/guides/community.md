<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Intéragisser avec les discussions et les pull requests

La librairie `huggingface_hub` fournir une interface Python pour intéragir avec les discussions et les pull requests du Hub.
Consultez [la page de documentation dédiée](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
pour un aperçu plus détaillé de ce que les discussions et les pull requests sur le Hub sont, et comment celles ci
fonctionnent en arrière plan.

## Récupérer les discussions et les pull requests depuis le Hub

La classe `HfApi` vous permet de récupérer des discussions et des pulls requests d'un dépôt en particulier:

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

`HfApi.get_repo_discussions` vous permet de filtrer par auteur, type (pull request ou discussion), et statut (`open` ou `closed`):


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

`HfApi.get_repo_discussions` renvoie un [générateur](https://docs.python.org/3.7/howto/functional.html#generators) qui prend
en charge des objets [`Discussion`]. Pour avoir toutes les discussions dans une seul liste, lancez: 

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

L'objet [`Discussion`] retourné par [`HfApi.get_repo_discussions`] contient une vue d'ensemble de la discussion
ou la pull request. Vous pouvez aussi obtenir des informations plus détaillée en utilisant [`HfApi.get_discussion_details`]:

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

[`HfApi.get_discussion_details`] renvoie un objet [`DiscuccionWithDetails`], qui est une sous-classe de [`Discussion`]
avec des informations plus détaillées sur la discussion ou la pull request. Les informations incluent tous les commentaires,
les changements de statut, et les changements de nom de la discussion via [`DiscussionWithDetails.events`].

En cas de pull request, vous pouvez récupérer la différence des versions git avec [`DiscussionWithDetails.diff`]. Tous les
commits de la pull request sont listés dans [`DiscussionWithDetails.events`].


## Créer et changer une discussion ou une pull request par le code

La classe [`HfApi`] fournit aussi des manière de créer et d'éditer des discussions et 
des pull requests. Vous aurez besoin d'un [token d'authentification](https://huggingface.co/docs/hub/security-tokens)
pour créer et modifier des discussions ou des pull requests.

La manière la plus simple de proposer des changements sur un dépôt du Hub est d'utiliser l'API [`create_commit`]:
fixez simplement le paramètre `create_pr` à `True`. Ce paramètre est aussi disponible avec d'autres méthodes
autour de [`create_commit`] telles que:

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

Vous pouvez aussi utiliser [`HfApi.create_discussion`] (respectivement [`hfApi.create_pull_request`]) pour créer une discussion (respectivement une pull
request) sur un dépôt. Ouvrir une pull request de cette manière peut-être utile si vous avez besoin de travailler sur des changements en local. Les
pull requests ouvertes de cette manière seront en mode `"draft"`.

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

La gestion des pull requests et des discussions peut être réalisée entièrement avec la classe [`HfApi`]. Par exemple:

    * [`comment_discussion`] to add comments
    * [`edit_discussion_comment`] to edit comments
    * [`rename_discussion`] to rename a Discussion or Pull Request 
    * [`change_discussion_status`] to open or close a Discussion / Pull Request 
    * [`merge_pull_request`] to merge a Pull Request 


Consultez la page de documentation [`HfApi`] pour une référence exhaustive de toutes les méthodes disponibles.

## Push les changement vers une pull request

*Arrive bientôt !*

## Voir aussi

Pour des références plus détaillées, consultez les pages de documentation [Discussions and Pull Requests](../package_reference/community) et [hf_api](../package_reference/hf_api).
