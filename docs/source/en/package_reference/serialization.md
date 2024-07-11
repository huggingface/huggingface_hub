<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Serialization

`huggingface_hub` contains helpers to help ML libraries serialize models weights in a standardized way. This part of the lib is still under development and will be improved in future releases. The goal is to harmonize how weights are serialized on the Hub, both to remove code duplication across libraries and to foster conventions on the Hub.

## Save torch state dict

The main helper of the `serialization` module takes a torch `nn.Module` as input and saves it to disk. It handles the logic to save shared tensors (see [safetensors explanation](https://huggingface.co/docs/safetensors/torch_shared_tensors)) as well as logic to split the state dictionary into shards, using [`split_torch_state_dict_into_shards`] under the hood. At the moment, only `torch` framework is supported.

If you want to save a state dictionary (e.g. a mapping between layer names and related tensors) instead of a `nn.Module`, you can use [`save_torch_state_dict`] which provides the same features. This is useful for example if you want to apply custom logic to the state dict before saving it.

[[autodoc]] huggingface_hub.save_torch_model

[[autodoc]] huggingface_hub.save_torch_state_dict

## Split state dict into shards

The `serialization` module also contains low-level helpers to split a state dictionary into several shards, while creating a proper index in the process. These helpers are available for `torch` and `tensorflow` tensors and are designed to be easily extended to any other ML frameworks.

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

### get_torch_storage_size

[[autodoc]] huggingface_hub.get_torch_storage_size