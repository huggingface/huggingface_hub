<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Télécharger des fichiers

## Télécharger un seul fichier

### hf_hub_download

[[autodoc]] huggingface_hub.hf_hub_download

### hf_hub_url

[[autodoc]] huggingface_hub.hf_hub_url

## Télécharger un snapshot du dépôt

[[autodoc]] huggingface_hub.snapshot_download

## Obtenir les métadonnées d'un fichier

### get_hf_file_metadata

[[autodoc]] huggingface_hub.get_hf_file_metadata

### HfFileMetadata

[[autodoc]] huggingface_hub.HfFileMetadata

## Mise en cache

Les méthodes affichées ci-dessus sont conçues pour fonctionner avec un système de mise en cache qui empêche
de retélécharger les fichiers. Le système de mise en cache a été mis à jour dans v0.8.0 pour devenir le système
de cache central partagé entre les bibliothèques qui dépendent du Hub.

Lisez le [guide du système de cache](../guides/manage-cache) pour une présentation détaillée de la mise en cache chez HF.
