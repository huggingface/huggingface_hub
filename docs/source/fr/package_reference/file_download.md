<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Téléchargement des fichiers

## Télécharger un unique fichier

### hf_hub_download

[[autodoc]] huggingface_hub.hf_hub_download

### hf_hub_url

[[autodoc]] huggingface_hub.hf_hub_url

## Télécharger un instantané du dépôt

[[autodoc]] huggingface_hub.snapshot_download

## Obtenir une métadonnée sur un fichier

### get_hf_file_metadata

[[autodoc]] huggingface_hub.get_hf_file_metadata

### HfFileMetadata

[[autodoc]] huggingface_hub.HfFileMetadata

## Utilisation du cache

Le méthodes affichées ci dessus sont faites pour fonctionner avec un système de cache
ce qui évite de retélécharger des fichiers. Le système de cache a été mis à jour dans
la version v0.8.0 afin d'être le système de cache central partagé dans toutes les
librairies dépendant du Hub.

Consultez le [guide cache-system](../guides/manage-cache) pour une présentation détaillée du caching à HF.