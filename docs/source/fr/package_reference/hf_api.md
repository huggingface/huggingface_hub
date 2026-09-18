<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# HfApi Client

Ci-dessous se trouve la documentation pour la classe `HfApi`, qui sert de wrapper Python pour l'API du Hugging Face Hub.

Toutes les méthodes de `HfApi` sont également accessibles directement depuis la racine du package. Les deux approches sont détaillées ci-dessous.

L'utilisation de la méthode racine est plus simple mais la classe [`HfApi`] vous offre plus de flexibilité.
En particulier, vous pouvez passer un token qui sera réutilisé dans tous les appels HTTP. Cela diffère
de `hf auth login` ou [`login`] car le token n'est pas persisté sur la machine.
Il est également possible de fournir un endpoint différent ou de configurer un user-agent personnalisé.

```python
from huggingface_hub import HfApi, list_models

# Utiliser la méthode racine
models = list_models()

# Ou configurer un client HfApi
hf_api = HfApi(
    endpoint="https://huggingface.co", # Peut être un endpoint Private Hub.
    token="hf_xxx", # Le token n'est pas persisté sur la machine.
)
models = hf_api.list_models()
```

## HfApi

[[autodoc]] HfApi

## API Dataclasses

### AccessRequest

[[autodoc]] huggingface_hub.hf_api.AccessRequest

### CommitInfo

[[autodoc]] huggingface_hub.hf_api.CommitInfo

### DatasetInfo

[[autodoc]] huggingface_hub.hf_api.DatasetInfo

### DryRunFileInfo

[[autodoc]] huggingface_hub.hf_api.DryRunFileInfo

### GitRefInfo

[[autodoc]] huggingface_hub.hf_api.GitRefInfo

### GitCommitInfo

[[autodoc]] huggingface_hub.hf_api.GitCommitInfo

### GitRefs

[[autodoc]] huggingface_hub.hf_api.GitRefs

### InferenceProviderMapping

[[autodoc]] huggingface_hub.hf_api.InferenceProviderMapping

### LFSFileInfo

[[autodoc]] huggingface_hub.hf_api.LFSFileInfo

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

### WebhookInfo

[[autodoc]] huggingface_hub.hf_api.WebhookInfo

### WebhookWatchedItem

[[autodoc]] huggingface_hub.hf_api.WebhookWatchedItem

## CommitOperation

Ci-dessous se trouvent les valeurs supportées pour [`CommitOperation`] :

[[autodoc]] CommitOperationAdd

[[autodoc]] CommitOperationDelete

[[autodoc]] CommitOperationCopy

## CommitScheduler

[[autodoc]] CommitScheduler
