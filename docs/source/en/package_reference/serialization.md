<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Serialization

`huggingface_hub` contains helpers to help ML libraries serialize models weights in a standardized way. This part of the lib is still under development and will be improved in future releases. The goal is to harmonize how weights are serialized on the Hub, both to remove code duplication across libraries and to foster conventions on the Hub.

## DDUF file format

DDUF is a file format designed for diffusers models. It allows saving all the information to run a model in a single file. This work is inspired by the GGUF format. `huggingface_hub` provides helpers to save and load DDUF files, ensuring the file format is respected.

<Tip warning={true}>

This is a very early version of the parser. The API and implementation can evolve in the near future.

The parser currently does very little validation. For more details about the file format, check out https://github.com/huggingface/huggingface.js/tree/main/packages/dduf.

</Tip>

### How to write a DDUF file?

Here is how to export a folder containing different parts of a diffusion model:

```python
# Export a folder as a DDUF file
>>> from huggingface_hub import export_folder_as_dduf
>>> export_folder_as_dduf("FLUX.1-dev.dduf", diffuser_path="path/to/FLUX.1-dev")
```

If your model is loaded in memory, you can directly serialize it to a GGUF file without saving to disk first.

```python
```

### How to read a DDUF file?

```python
>>> import json
>>> import safetensors.load
>>> from huggingface_hub import read_dduf_file

# Read DDUF metadata
>>> dduf_entries = read_dduf_file("FLUX.1-dev.dduf")

# Returns a mapping filename <> DDUFEntry
>>> dduf_entries["model_index.json"]
DDUFEntry(filename='model_index.json', offset=66, length=587)

# Load model index as JSON
>>> json.loads(dduf_entries["model_index.json"].read_text())
{'_class_name': 'FluxPipeline', '_diffusers_version': '0.32.0.dev0', '_name_or_path': 'black-forest-labs/FLUX.1-dev', 'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'], 'text_encoder': ['transformers', 'CLIPTextModel'], 'text_encoder_2': ['transformers', 'T5EncoderModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'], 'tokenizer_2': ['transformers', 'T5TokenizerFast'], 'transformer': ['diffusers', 'FluxTransformer2DModel'], 'vae': ['diffusers', 'AutoencoderKL']}

# Load VAE weights using safetensors
>>> with dduf_entries["vae/diffusion_pytorch_model.safetensors"].as_mmap() as mm:
...     state_dict = safetensors.torch.load(mm)
```

### Helpers

[[autodoc]] huggingface_hub.export_as_dduf

[[autodoc]] huggingface_hub.export_folder_as_dduf

[[autodoc]] huggingface_hub.read_dduf_file

[[autodoc]] huggingface_hub.DDUFEntry

### Errors

[[autodoc]] huggingface_hub.errors.DDUFError

[[autodoc]] huggingface_hub.errors.DDUFCorruptedFileError

[[autodoc]] huggingface_hub.errors.DDUFExportError

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