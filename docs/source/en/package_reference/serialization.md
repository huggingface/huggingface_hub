<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Serialization

`huggingface_hub` provides helpers to save and load ML model weights in a standardized way. This part of the library is still under development and will be improved in future releases. The goal is to harmonize how weights are saved and loaded across the Hub, both to remove code duplication across libraries and to establish consistent conventions.

## DDUF file format

DDUF is a file format designed for diffusion models. It allows saving all the information to run a model in a single file. This work is inspired by the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format. `huggingface_hub` provides helpers to save and load DDUF files, ensuring the file format is respected.

> [!WARNING]
> This is a very early version of the parser. The API and implementation can evolve in the near future.
>
> The parser currently does very little validation. For more details about the file format, check out https://github.com/huggingface/huggingface.js/tree/main/packages/dduf.

### How to write a DDUF file?

Here is how to export a folder containing different parts of a diffusion model using [`export_folder_as_dduf`]:

```python
# Export a folder as a DDUF file
>>> from huggingface_hub import export_folder_as_dduf
>>> export_folder_as_dduf("FLUX.1-dev.dduf", folder_path="path/to/FLUX.1-dev")
```

For more flexibility, you can use [`export_entries_as_dduf`] and pass a list of files to include in the final DDUF file:

```python
# Export specific files from the local disk.
>>> from huggingface_hub import export_entries_as_dduf
>>> export_entries_as_dduf(
...     dduf_path="stable-diffusion-v1-4-FP16.dduf",
...     entries=[ # List entries to add to the DDUF file (here, only FP16 weights)
...         ("model_index.json", "path/to/model_index.json"),
...         ("vae/config.json", "path/to/vae/config.json"),
...         ("vae/diffusion_pytorch_model.fp16.safetensors", "path/to/vae/diffusion_pytorch_model.fp16.safetensors"),
...         ("text_encoder/config.json", "path/to/text_encoder/config.json"),
...         ("text_encoder/model.fp16.safetensors", "path/to/text_encoder/model.fp16.safetensors"),
...         # ... add more entries here
...     ]
... )
```

The `entries` parameter also supports passing an iterable of paths or bytes. This can prove useful if you have a loaded model and want to serialize it directly into a DDUF file instead of having to serialize each component to disk first and then as a DDUF file. Here is an example of how a `StableDiffusionPipeline` can be serialized as DDUF:


```python
# Export state_dicts one by one from a loaded pipeline 
>>> from diffusers import DiffusionPipeline
>>> from typing import Generator, Tuple
>>> import safetensors.torch
>>> from huggingface_hub import export_entries_as_dduf
>>> pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
... # ... do some work with the pipeline

>>> def as_entries(pipe: DiffusionPipeline) -> Generator[Tuple[str, bytes], None, None]:
...     # Build a generator that yields the entries to add to the DDUF file.
...     # The first element of the tuple is the filename in the DDUF archive (must use UNIX separator!). The second element is the content of the file.
...     # Entries will be evaluated lazily when the DDUF file is created (only 1 entry is loaded in memory at a time)
...     yield "vae/config.json", pipe.vae.to_json_string().encode()
...     yield "vae/diffusion_pytorch_model.safetensors", safetensors.torch.save(pipe.vae.state_dict())
...     yield "text_encoder/config.json", pipe.text_encoder.config.to_json_string().encode()
...     yield "text_encoder/model.safetensors", safetensors.torch.save(pipe.text_encoder.state_dict())
...     # ... add more entries here

>>> export_entries_as_dduf(dduf_path="stable-diffusion-v1-4.dduf", entries=as_entries(pipe))
```

**Note:** in practice, `diffusers` provides a method to directly serialize a pipeline in a DDUF file. The snippet above is only meant as an example.

### How to read a DDUF file?

```python
>>> import json
>>> import safetensors.torch
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

[[autodoc]] huggingface_hub.export_entries_as_dduf

[[autodoc]] huggingface_hub.export_folder_as_dduf

[[autodoc]] huggingface_hub.read_dduf_file

[[autodoc]] huggingface_hub.DDUFEntry

### Errors

[[autodoc]] huggingface_hub.errors.DDUFError

[[autodoc]] huggingface_hub.errors.DDUFCorruptedFileError

[[autodoc]] huggingface_hub.errors.DDUFExportError

[[autodoc]] huggingface_hub.errors.DDUFInvalidEntryNameError

## Saving tensors

The main helper of the `serialization` module takes a torch `nn.Module` as input and saves it to disk. It handles the logic to save shared tensors (see [safetensors explanation](https://huggingface.co/docs/safetensors/torch_shared_tensors)) as well as logic to split the state dictionary into shards, using [`split_torch_state_dict_into_shards`] under the hood. At the moment, only `torch` framework is supported.

If you want to save a state dictionary (e.g. a mapping between layer names and related tensors) instead of a `nn.Module`, you can use [`save_torch_state_dict`] which provides the same features. This is useful for example if you want to apply custom logic to the state dict before saving it.

### save_torch_model

[[autodoc]] huggingface_hub.save_torch_model

### save_torch_state_dict

[[autodoc]] huggingface_hub.save_torch_state_dict


The `serialization` module also contains low-level helpers to split a state dictionary into several shards, while creating a proper index in the process. These helpers are available for `torch` tensors and are designed to be easily extended to any other ML frameworks.

### split_torch_state_dict_into_shards

[[autodoc]] huggingface_hub.split_torch_state_dict_into_shards

### split_state_dict_into_shards_factory

This is the underlying factory from which each framework-specific helper is derived. In practice, you are not expected to use this factory directly except if you need to adapt it to a framework that is not yet supported. If that is the case, please let us know by [opening a new issue](https://github.com/huggingface/huggingface_hub/issues/new) on the `huggingface_hub` repo.

[[autodoc]] huggingface_hub.split_state_dict_into_shards_factory

## Loading tensors

The loading helpers support both single-file and sharded checkpoints in either safetensors or pickle format. [`load_torch_model`] takes a `nn.Module` and a checkpoint path (either a single file or a directory) as input and load the weights into the model.

### load_torch_model

[[autodoc]] huggingface_hub.load_torch_model

### load_state_dict_from_file

[[autodoc]] huggingface_hub.load_state_dict_from_file

## Tensors helpers

### get_torch_storage_id

[[autodoc]] huggingface_hub.get_torch_storage_id

### get_torch_storage_size

[[autodoc]] huggingface_hub.get_torch_storage_size