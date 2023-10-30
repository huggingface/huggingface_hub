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

Two of these parameters are intuitive (`author` and `search`), but what about that `filter`?
`filter` takes as input a [`ModelFilter`] object (or [`DatasetFilter`]). You can instantiate
it by specifying which models you want to filter. 

Let's see an example to get all models on the Hub that does image classification, have been
trained on the imagenet dataset and that runs with PyTorch. That can be done with a single
[`ModelFilter`]. Attributes are combined as "logical AND".

```py
models = hf_api.list_models(
    filter=ModelFilter(
		task="image-classification",
		library="pytorch",
		trained_dataset="imagenet"
	)
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
	(...)
```


## How to explore filter options ?

Now you know how to filter your list of models/datasets/spaces. The problem you might
have is that you don't know exactly what you are looking for. No worries! We also provide
some helpers that allows you to discover what arguments can be passed in your query.

[`ModelSearchArguments`] and [`DatasetSearchArguments`] are nested namespace objects that
have **every single option** available on the Hub and that will return what should be passed
to `filter`. The best of all is: it has tab completion 🎊 .

```python
>>> from huggingface_hub import ModelSearchArguments, DatasetSearchArguments

>>> model_args = ModelSearchArguments()
>>> dataset_args = DatasetSearchArguments()
```

<Tip warning={true}>

Before continuing, please we aware that [`ModelSearchArguments`] and [`DatasetSearchArguments`]
are legacy helpers meant for exploratory purposes only. Their initialization require listing
all models and datasets on the Hub which makes them increasingly slower as the number of repos
on the Hub increases. For some production-ready code, consider passing raw strings when making
a filtered search on the Hub.

</Tip>

Now, let's check what is available in `model_args` by checking it's output, you will find:

```python
>>> model_args
Available Attributes or Keys:
 * author
 * dataset
 * language
 * library
 * license
 * model_name
 * pipeline_tag
```

It has a variety of attributes or keys available to you. This is because it is both an object
and a dictionary, so you can either do `model_args["author"]` or `model_args.author`.

The first criteria is getting all PyTorch models. This would be found under the `library` attribute, so let's see if it is there:

```python
>>> model_args.library
Available Attributes or Keys:
 * AdapterTransformers
 * Asteroid
 * ESPnet
 * Fairseq
 * Flair
 * JAX
 * Joblib
 * Keras
 * ONNX
 * PyTorch
 * Rust
 * Scikit_learn
 * SentenceTransformers
 * Stable_Baselines3 (Key only)
 * Stanza
 * TFLite
 * TensorBoard
 * TensorFlow
 * TensorFlowTTS
 * Timm
 * Transformers
 * allenNLP
 * fastText
 * fastai
 * pyannote_audio
 * spaCy
 * speechbrain
```

It is! The `PyTorch` name is there, so you'll need to use `model_args.library.PyTorch`:

```python
>>> model_args.library.PyTorch
'pytorch'
```

Below is an animation repeating the process for finding both the `Text Classification` and `glue` requirements:

![Animation exploring `model_args.pipeline_tag`](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/search_text_classification.gif)

![Animation exploring `model_args.dataset`](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/search_glue.gif)

Now that all the pieces are there, the last step is to combine them all for something the
API can use through the [`ModelFilter`] and [`DatasetFilter`] classes (i.e. strings).


```python
>>> from huggingface_hub import ModelFilter, DatasetFilter

>>> filt = ModelFilter(
...     task=model_args.pipeline_tag.TextClassification, 
...     trained_dataset=dataset_args.dataset_name.glue, 
...     library=model_args.library.PyTorch
... )
>>> list(api.list_models(filter=filt))[0]
ModelInfo(
	id='Graphcore/gptj-mnli',
	author='Graphcore',
	sha='567b3e0fb7353bb6c7de3613d2e2c650e9b3e559',
	last_modified=datetime.datetime(2022, 8, 25, 11, 39, 23, tzinfo=datetime.timezone.utc),
	private=False,
	gated=None, 
	disabled=None,
	downloads=0,
	likes=1,
	library_name='transformers',
	tags=['transformers', 'pytorch', 'gptj', 'text-generation', 'causal-lm', 'text-classification', 'en', 'dataset:glue', 'arxiv:1910.10683', 'arxiv:2104.09864', 'license:apache-2.0', 'model-index', 'endpoints_compatible', 'region:us'], 
	pipeline_tag='text-generation',
	(...)
)
```

As you can see, it found the models that fit all the criteria. You can even take it further
by passing in an array for each of the parameters from before. For example, let's take a look
for the same configuration, but also include `TensorFlow` in the filter:


```python
>>> filt = ModelFilter(
...     task=model_args.pipeline_tag.TextClassification, 
...     library=[model_args.library.PyTorch, model_args.library.TensorFlow]
... )
>>> list(list_models(filter=filt))[0]
ModelInfo(
	id='Jiva/xlm-roberta-large-it-mnli',
	author='Jiva',
	sha='b4491a90367f80cc6a2b225a7fe16a501ad301df',
	last_modified=datetime.datetime(2023, 5, 22, 9, 22, 29, tzinfo=datetime.timezone.utc),
	private=False,
	gated=None,
	disabled=None,
	downloads=954,
	likes=6,
	library_name='transformers',
	tags=['transformers', 'pytorch', 'safetensors', 'xlm-roberta', 'text-classification', 'tensorflow', 'zero-shot-classification', 'it', 'dataset:multi_nli', 'dataset:glue', 'arxiv:1911.02116', 'license:mit', 'model-index', 'endpoints_compatible', 'has_space', 'region:us'],
	pipeline_tag='zero-shot-classification', 
	(...)
)
```

This query is strictly equivalent to:

```py
>>> filt = ModelFilter(
...     task="text-classification", 
...     library=["pytorch", "tensorflow"],
... )
```

Here, the [`ModelSearchArguments`] has been a helper to explore the options available on the Hub.
However, it is not a requirement to make a search. Another way to do that is to visit the
[models](https://huggingface.co/models) and [datasets](https://huggingface.co/datasets) pages
in your browser, search for some parameters and look at the values in the URL.

