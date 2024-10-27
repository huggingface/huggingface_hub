<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interaktion mit Diskussionen und Pull-Requests

Die `huggingface_hub`-Bibliothek bietet eine Python-Schnittstelle, um mit Pull-Requests und Diskussionen auf dem Hub zu interagieren. Besuchen Sie [die spezielle Dokumentationsseite](https://huggingface.co/docs/hub/repositories-pull-requests-discussions), um einen tieferen Einblick in Diskussionen und Pull-Requests auf dem Hub zu erhalten und zu erfahren, wie sie im Hintergrund funktionieren.

## Diskussionen und Pull-Requests vom Hub abrufen

Die Klasse `HfApi` ermöglicht es Ihnen, Diskussionen und Pull-Requests zu einem gegebenen Repository abzurufen:

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(repo_id="bigscience/bloom-1b3"):
...     print(f"{discussion.num} - {discussion.title}, pr: {discussion.is_pull_request}")

# 11 - Add Flax weights, pr: True
# 10 - Update README.md, pr: True
# 9 - Training languages in the model card, pr: True
# 8 - Update tokenizer_config.json, pr: True
# 7 - Slurm training script, pr: False
[...]
```

`HfApi.get_repo_discussions` gibt einen [Generator](https://docs.python.org/3.7/howto/functional.html#generators) zurück, der [`Diskussion`]-Objekte liefert. Um alle Diskussionen in einer einzelnen Liste zu erhalten, führen Sie den folgenden Befehl aus:

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

Das von [`HfApi.get_repo_discussions`] zurückgegebene [`Diskussion`]-Objekt enthält einen Überblick über die Diskussion oder Pull-Requests. Sie können auch detailliertere Informationen mit [`HfApi.get_discussion_details`] abrufen:

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

[`HfApi.get_discussion_details`] gibt ein [`DiskussionMitDetails`]-Objekt zurück, das eine Unterklasse von [`Diskussion`] mit detaillierteren Informationen über die Diskussion oder Pull-Requests ist. Informationen beinhalten alle Kommentare, Statusänderungen und Umbenennungen der Diskussion mittels [`DiskussionMitDetails.events`].

Im Fall eines Pull-Requests können Sie mit [`DiskussionMitDetails.diff`] den rohen git diff abrufen. Alle Commits des Pull-Requests sind in [`DiskussionMitDetails.events`] aufgelistet.


## Diskussion oder Pull-Request programmatisch erstellen und bearbeiten

Die [`HfApi`]-Klasse bietet auch Möglichkeiten, Diskussionen und Pull-Requests zu erstellen und zu bearbeiten. Sie benötigen ein [Access Token](https://huggingface.co/docs/hub/security-tokens), um Diskussionen oder Pull-Requests zu erstellen und zu bearbeiten.

Die einfachste Möglichkeit, Änderungen an einem Repo auf dem Hub vorzuschlagen, ist über die [`create_commit`]-API: Setzen Sie einfach das `create_pr`-Parameter auf `True`. Dieser Parameter ist auch bei anderen Methoden verfügbar, die [`create_commit`] umfassen:

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

Sie können auch [`HfApi.create_discussion`] (bzw. [`HfApi.create_pull_request`]) verwenden, um eine Diskussion (bzw. einen Pull-Request) für ein Repository zu erstellen. Das Öffnen eines Pull-Requests auf diese Weise kann nützlich sein, wenn Sie lokal an Änderungen arbeiten müssen. Auf diese Weise geöffnete Pull-Requests befinden sich im `"Entwurfs"`-Modus.

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

Das Verwalten von Pull-Requests und Diskussionen kann vollständig mit der [`HfApi`]-Klasse durchgeführt werden. Zum Beispiel:

    * [`comment_discussion`] zum Hinzufügen von Kommentaren
    * [`edit_discussion_comment`] zum Bearbeiten von Kommentaren
    * [`rename_discussion`] zum Umbenennen einer Diskussion oder eines Pull-Requests
    * [`change_discussion_status`] zum Öffnen oder Schließen einer Diskussion / eines Pull-Requests
    * [`merge_pull_request`] zum Zusammenführen eines Pull-Requests

Besuchen Sie die [`HfApi`]-Dokumentationsseite für eine vollständige Übersicht aller verfügbaren Methoden.

## Änderungen an einen Pull-Request senden

*Demnächst verfügbar !*

## Siehe auch

Für eine detailliertere Referenz besuchen Sie die [Diskussionen und Pull-Requests](../package_reference/community) und die [hf_api](../package_reference/hf_api)-Dokumentationen.
