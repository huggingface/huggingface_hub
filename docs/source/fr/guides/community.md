<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interagir avec les Discussions et Pull Requests

La bibliothèque `huggingface_hub` fournit une interface Python pour interagir avec les Pull Requests et Discussions sur le Hub. Consultez [la page de documentation dédiée](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) pour avoir une vue plus approfondie de ce que sont les Discussions, les Pull Requests, et comment elles fonctionnent en interne.

## Récupérer les Discussions et Pull Requests du Hub

La classe `HfApi` vous permet de récupérer les Discussions et Pull Requests sur un dépôt donné :

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

`HfApi.get_repo_discussions` prend en charge le filtrage par auteur, type (Pull Request ou Discussion) et statut (`open` ou `closed`) :

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

`HfApi.get_repo_discussions` retourne un [générateur](https://docs.python.org/3.7/howto/functional.html#generators) qui produit des objets [`Discussion`]. Pour obtenir toutes les Discussions dans une seule liste, exécutez :

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

L'objet [`Discussion`] retourné par [`HfApi.get_repo_discussions`] contient un aperçu de la Discussion ou de la Pull Request. Vous pouvez également obtenir des informations plus détaillées en utilisant [`HfApi.get_discussion_details`] :

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
    diff='diff --git a/README.md b/README.md\nindex a6ae3b9294edf8d0eda0d67c7780a10241242a7e..3a1814f212bc3f0d3cc8f74bdbd316de4ae7b9e3 100644\n--- a/README.md\n+++ b/README.md\n@@ -132,7 +132 [...]',
)
```

[`HfApi.get_discussion_details`] retourne un objet [`DiscussionWithDetails`], qui est une sous-classe de [`Discussion`] avec des informations plus détaillées sur la Discussion ou Pull Request. Les informations incluent tous les commentaires, changements de statut et renommages de la Discussion via [`DiscussionWithDetails.events`].

Dans le cas d'une Pull Request, vous pouvez récupérer le diff git brut avec [`DiscussionWithDetails.diff`]. Tous les commits de la Pull Request sont listés dans [`DiscussionWithDetails.events`].

## Créer et éditer une Discussion ou Pull Request par programme

La classe [`HfApi`] offre également des moyens de créer et d'éditer des Discussions et Pull Requests. Vous aurez besoin d'un [access token](https://huggingface.co/docs/hub/security-tokens) pour créer et éditer des Discussions ou Pull Requests.

Le moyen le plus simple de proposer des changements sur un dépôt du Hub est via l'API [`create_commit`] : il suffit de définir le paramètre `create_pr` à `True`. Ce paramètre est également disponible sur d'autres méthodes qui encapsulent [`create_commit`] :

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

Vous pouvez également utiliser [`HfApi.create_discussion`] (respectivement [`HfApi.create_pull_request`]) pour créer une Discussion (une Pull Request) sur un dépôt. Ouvrir une Pull Request de cette manière peut être utile si vous devez travailler sur des changements local. Les Pull Requests ouvertes de cette manière seront en mode `"draft"` par défaut.

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

La gestion des Pull Requests et Discussions peut être effectuée entièrement avec la classe [`HfApi`]. Par exemple :

* [`comment_discussion`] pour ajouter des commentaires
* [`edit_discussion_comment`] pour éditer des commentaires
* [`rename_discussion`] pour renommer une Discussion ou Pull Request
* [`change_discussion_status`] pour ouvrir ou fermer une Discussion / Pull Request
* [`merge_pull_request`] pour merger une Pull Request

Consultez la page de documentation [`HfApi`] pour une référence exhaustive de toutes les méthodes disponibles.

## Pousser des changements vers une Pull Request

*Prochainement !*

## Voir aussi

Pour une référence plus détaillée, consultez les pages de documentation [Discussions et Pull Requests](../package_reference/community) et [hf_api](../package_reference/hf_api).
