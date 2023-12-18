<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# HfApi Client

Ci dessous la documentation pour la classe `HfApi`, qui sert de wrapper Python pour l'API Hugging Face Hub.

Toutes les méthodes de `HfApi` sont aussi accessibles depuis la racine du package directement. Les deux approches sont détaillées ci-dessous.

Utiliser la méthode du chemin racine est plus direct mais la classe [`HfApi`] donne plus de flexibilité.
En particulier,  vous pouvez mettre un token qui va être réutilisé dans tous les appels HTTP. C'est
différent de la commande `huggingface-cli login` ou [`login`] vu que le token n'est pas enregistré
sur la machine. Il est aussi possible de fournir un endpoint différent ou de configurer un user-agent
personnalisé.

```python
from huggingface_hub import HfApi, list_models

# Utilisez la méthode du chemin racine
models = list_models()

# Ou configurez le client HfApi
hf_api = HfApi(
    endpoint="https://huggingface.co", # Vous pouvez mettre un endpoint de Hub privéC.
    token="hf_xxx", # Le token n'est pas sauvegardé sur la machine.
)
models = hf_api.list_models()
```

## HfApi

[[autodoc]] HfApi

[[autodoc]] plan_multi_commits

## API Dataclasses

### CommitInfo

[[autodoc]] huggingface_hub.hf_api.CommitInfo

### DatasetInfo

[[autodoc]] huggingface_hub.hf_api.DatasetInfo

### GitRefInfo

[[autodoc]] huggingface_hub.hf_api.GitRefInfo

### GitCommitInfo

[[autodoc]] huggingface_hub.hf_api.GitCommitInfo

### GitRefs

[[autodoc]] huggingface_hub.hf_api.GitRefs

### ModelInfo

[[autodoc]] huggingface_hub.hf_api.ModelInfo

### RepoSibling

[[autodoc]] huggingface_hub.hf_api.RepoSibling

### RepoFile

[[autodoc]] huggingface_hub.hf_api.RepoFile

### RepoUrl

[[autodoc]] huggingface_hub.hf_api.RepoUrl

### SafetensorsRepoMetadata

[[autodoc]] huggingface_hub.utils.SafetensorsRepoMetadata

### SafetensorsFileMetadata

[[autodoc]] huggingface_hub.utils.SafetensorsFileMetadata

### SpaceInfo

[[autodoc]] huggingface_hub.hf_api.SpaceInfo

### TensorInfo

[[autodoc]] huggingface_hub.utils.TensorInfo

### User

[[autodoc]] huggingface_hub.hf_api.User

### UserLikes

[[autodoc]] huggingface_hub.hf_api.UserLikes

## CommitOperation

Ci dessous les valeurs supportés pour [`CommitOperation`]:

[[autodoc]] CommitOperationAdd

[[autodoc]] CommitOperationDelete

[[autodoc]] CommitOperationCopy

## CommitScheduler

[[autodoc]] CommitScheduler

## Token helper

`huggingface_hub` garde en mémoire l'information d'authentification en local pour qu'il puisse être réutilisé
dans les méthodes suivantes.

La librairie fait ceci en utilisant l'utilitaire [`HfFolder`], qui sauvegarde de la donnée
à la racine de l'utilisateur.

[[autodoc]] HfFolder

## Search helpers

Certains helpers pour filtrer les dépôts sur le Hub sont disponibles dans le package
`huggingface_hub`.

### DatasetFilter

[[autodoc]] DatasetFilter

### ModelFilter

[[autodoc]] ModelFilter

