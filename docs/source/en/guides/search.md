<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
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
- ...

Let's see an example to get all models on the Hub that does image classification, have been trained on the imagenet dataset and that runs with PyTorch.

```py
models = hf_api.list_models(
	task="image-classification",
	library="pytorch",
	trained_dataset="imagenet",
)
```

While filtering, you can also sort the models and take only the top results. For example,
the following example fetches the top 5 most downloaded datasets on the Hub:

```py
>>> list(list_datasets(sort="downloads", direction=-1, limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



To explore available filter on the Hub, visit [models](https://huggingface.co/models) and [datasets](https://huggingface.co/datasets) pages
in your browser, search for some parameters and look at the values in the URL.
