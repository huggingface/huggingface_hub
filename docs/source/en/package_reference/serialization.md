<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Serialization

`huggingface_hub` contains helpers to help ML libraries to serialize models weights in a standardized way. This part of the lib is still under development and will be improved in future releases. The goal is to harmonize how weights are serialized on the Hub, both to remove code duplication across libraries and to foster conventions on the Hub.

## Split state dict into shards

At the moment, this module contains a single helper that takes a state dictionary (e.g. a mapping between layer names and related tensors) and split it into several shards, while creating a proper index in the process. This helper is available for `torch`, `tensorflow` and `numpy` tensors and is designed to be easily extended to any other ML frameworks.

### split_numpy_state_dict_into_shards

[[autodoc]] huggingface_hub.split_numpy_state_dict_into_shards

### split_tf_state_dict_into_shards

[[autodoc]] huggingface_hub.split_tf_state_dict_into_shards

### split_torch_state_dict_into_shards

[[autodoc]] huggingface_hub.split_torch_state_dict_into_shards

### split_state_dict_into_shards_factory

This is the underlying factory from which each framework-specific helper is derived. In practice, you are not expected to use this factory directly except if you need to adapt it to a framework that is not yet supported. If that is the case, please let us know by [opening a new issue](https://github.com/huggingface/huggingface_hub/issues/new) on the `huggingface_hub` repo.

[[autodoc]] huggingface_hub.split_state_dict_into_shards_factory

## Helpers

### get_torch_storage_id

[[autodoc]] huggingface_hub.get_torch_storage_id