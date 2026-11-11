<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Search the Hub

In this tutorial, you will learn how to search models, datasets and spaces on the Hub using `huggingface_hub`.

## How to list repositories ?

`huggingface_hub` library includes an HTTP client [`HfApi`] to interact with the Hub.
Among other things, it can list models, datasets and spaces stored on the Hub:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

The output of [`list_models`] is an iterator over the models stored on the Hub.

Similarly, you can use [`list_datasets`] to list datasets and [`list_spaces`] to list Spaces.

## How to filter repositories ?

Listing repositories is great but now you might want to filter your search.
The list helpers have several attributes like:
- `filter`
- `author`
- `search`
- `num_parameters`
- ...

Let's see an example to get all models on the Hub that does image classification, have been trained on the imagenet dataset and that runs with PyTorch.

```py
models = hf_api.list_models(filter=["image-classification", "pytorch", "imagenet"])
```

You can also filter models by parameter count using the same range syntax as the Hub UI:

```py
models = hf_api.list_models(num_parameters="min:6B,max:128B")
```

## How to filter by modality ?

`pipeline_tag` targets a single task, such as `depth-estimation` or `text-classification`. To browse a whole
family of tasks at once, use `modality` instead. For example, to get the trending audio models, whatever their
specific task:

```py
>>> api.list_models(modality="audio", sort="trending_score", limit=10)
```

The accepted modalities are `"text"`, `"image"`, `"audio"`, `"video"`, `"3d"`, `"tabular"`, `"multimodal"`,
`"document"`, `"timeseries"`, `"reinforcement-learning"` and `"other"` for models, and `"text"`, `"image"`,
`"audio"`, `"video"`, `"3d"`, `"tabular"`, `"document"`, `"timeseries"` and `"geospatial"` for datasets.
Aliases such as `"nlp"`, `"computer-vision"` or `"speech"` are accepted too.

```py
>>> api.list_datasets(modality="audio", sort="downloads", limit=10)
```

Modalities are broader than tags, so a modality filter can also be combined with the other filters
(`author`, `search`, `gated`, ...) but not with `pipeline_tag`, which is a single task.

While filtering, you can also sort the models and take only the top results. For example,
the following example fetches the top 5 most downloaded datasets on the Hub:

```py
>>> list(list_datasets(sort="downloads", limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



To explore available filters on the Hub, visit [models](https://huggingface.co/models) and [datasets](https://huggingface.co/datasets) pages
in your browser, search for some parameters and look at the values in the URL.

## Using the CLI

You can also list and search for models, datasets, and Spaces using the `hf` command-line interface:

```bash
# List models
>>> hf models ls --search "llama" --sort downloads --limit 5

# List datasets
>>> hf datasets ls --author Qwen

# List models or datasets of a whole modality
>>> hf models ls --modality audio --sort trending_score --limit 10
>>> hf datasets ls --modality audio --sort downloads --limit 10

# List Spaces
>>> hf spaces ls --search "3d"

# Get info about a specific model
>>> hf models info Lightricks/LTX-2
```

For more details, see the [CLI guide](./cli.md#hf-models).
