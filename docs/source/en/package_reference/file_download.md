<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Downloading files

## Download a single file

### hf_hub_download

[[autodoc]] huggingface_hub.hf_hub_download

### hf_hub_url

[[autodoc]] huggingface_hub.hf_hub_url

## Download a snapshot of the repo

[[autodoc]] huggingface_hub.snapshot_download

## Get metadata about a file

### get_hf_file_metadata

[[autodoc]] huggingface_hub.get_hf_file_metadata

### HfFileMetadata

[[autodoc]] huggingface_hub.HfFileMetadata

## Caching

The methods displayed above are designed to work with a caching system that prevents
re-downloading files. The caching system was updated in v0.8.0 to become the central
cache-system shared across libraries that depend on the Hub.

Read the [cache-system guide](../guides/manage-cache) for a detailed presentation of caching at
at HF.
