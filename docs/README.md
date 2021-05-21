---
title: Model Hub docs
---

<h1>Model Hub documentation</h1>


## What's the Hugging Face model hub?

We are helping the community work together towards the goal of advancing Artificial Intelligence 🔥.

Not one company, even the Tech Titans, will be able to “solve AI” by itself – the only way we'll achieve this is by sharing knowledge and resources. On this model hub we are building the largest collection of models, datasets and metrics to democratize and advance AI and NLP for everyone 🚀.

## When sharing a model, what should I add to my model card?

In your README.md model card you should:
- describe your model,
- its intended uses & potential limitations, including bias and ethical considerations as detailed in [[Mitchell, 2018]](https://arxiv.org/abs/1810.03993)
- your training params and experimental info – you can embed or link to an experiment tracking platform for reference
- which datasets did you train on, and your eval results.

If needed you can find a template [here](https://github.com/huggingface/model_card).


## What metadata can I add to my model card?

In addition to textual (markdown) content, to unlock helpful features you can add any or all of the following items to a [YAML](https://en.wikipedia.org/wiki/YAML) metadata block at the top of your model card:

```yaml
---
language: "ISO 639-1 code for your language, or `multilingual`"
thumbnail: "url to a thumbnail used in social sharing"
tags:
- array
- of
- tags
license: "any valid license identifier"
datasets:
- array of dataset identifiers
metrics:
- array of metric identifiers
---
```

License identifiers are those standardized by GitHub in the right column (keywords) [here](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#searching-github-by-license-type).

Dataset and metric identifiers are those listed on the [datasets](https://huggingface.co/datasets) and [metrics](https://huggingface.co/metrics) pages and in the [`datasets`](https://github.com/huggingface/datasets) repository.

All the tags can then be used to filter the list of models on https://huggingface.co/models.


## How are model tags determined?

On top of each model page (see e.g. [`distilbert-base-uncased`](/distilbert-base-uncased)) you'll see the model's tags – they help for discovery and condition which features are enabled on which model page.

- The weight files that compose the models condition the framework(s) like `pytorch`, `tf`, etc.
- The `"architectures"` field of the model's config.json file – which should be automatically filled if you save your model using `.save_pretrained()` – condition the type of pipeline used in the inference API, and the type of widget present on the model page
	- A simplified snapshot of the mapping code can be found in [this gist](https://gist.github.com/julien-c/857ba86a6c6a895ecd90e7f7cab48046).
- If your config.json file contains a `task_specific_params` subfield, its sub-keys will be added as `pipeline:` tags. All parameters defined under this sub-key will overwrite the default parameters in config.json when running the corresponding pipeline. See [`t5-base`](https://huggingface.co/t5-base) for example.
- Most other metadata from the metadata block are also added as extra tags, at the end of the list.


## How is a model's type of inference API and widget determined?

To determine which pipeline and widget to display (text-classification, token-classification, translation, etc.), we use a simple mapping from model tags to one particular `pipeline_tag` (we currently only expose *one* pipeline and widget on each model page, even for models that would support several).

We try to use the most specific pipeline for each model, see pseudo-code in [this gist](https://gist.github.com/julien-c/857ba86a6c6a895ecd90e7f7cab48046).

You can always manually override your pipeline type with `pipeline_tag: xxx` in your model card metadata.

## What are all the possible pipeline/widget types?

Here they are, with links to examples:
- `text-classification`, for instance [`roberta-large-mnli`](https://huggingface.co/roberta-large-mnli)
- `token-classification`, for instance [`dbmdz/bert-large-cased-finetuned-conll03-english`](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)
- `question-answering`, for instance [`distilbert-base-uncased-distilled-squad`](https://huggingface.co/distilbert-base-uncased-distilled-squad)
- `translation`, for instance [`t5-base`](https://huggingface.co/t5-base)
- `summarization`, for instance [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn)
- `conversational`, for instance [`facebook/blenderbot-400M-distill`](https://huggingface.co/facebook/blenderbot-400M-distill)
- `text-generation`, for instance [`gpt2`](https://huggingface.co/gpt2)
- `fill-mask`, for instance [`distilroberta-base`](https://huggingface.co/distilroberta-base)
- `zero-shot-classification` (implemented on top of a nli `text-classification` model), for instance [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)
- `table-question-answering`, for instance [`google/tapas-base-finetuned-wtq`](https://huggingface.co/google/tapas-base-finetuned-wtq)

## How can I control my model's widget's example inputs?

Example inputs are the random inputs that pre-populate your widget on page launch (unless you specify an input by URL parameters).

We try to provide example inputs for some languages and widget types, but it's better if you provide your own examples. You can add them to your model card: see [this commit](https://github.com/huggingface/transformers/commit/6a495cae0090307198131c07cd4f3f1e9b38b4e6) for the format you need to use.

If we don't provide default inputs for your model's language, please open a PR against [this DefaultWidget.ts file](https://github.com/huggingface/widgets-server/blob/master/DefaultWidget.ts) to add them. Thanks!


## How can I turn off the inference API for my model?

Specify `inference: false` in your model card's metadata.


## Can I send large volumes of requests? Can I get accelerated APIs?

If you are interested in accelerated inference and/or higher volumes of requests and/or a SLA, please contact us at `api-enterprise at huggingface.co`.


## What technology do you use to power the inference API?

The API is built on top of our [Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) feature.

On top of Pipelines and depending on the model type, we build a number of production optimizations like:
- compiling models to optimized intermediary representations (e.g. [ONNX](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)),
- maintaining a Least Recently Used cache ensuring that the most popular models are always loaded,
- scaling the underlying compute infrastructure on the fly depending on the load constraints.


## Can I write \\( \LaTeX \\) in my model card?

Yes, we use the [KaTeX](https://katex.org/) math typesetting library to render math formulas server-side,
before parsing the markdown.
You have to use the following delimiters:
- `$$ ... $$` for display mode
- `\\` `(` `...` `\\` `)` for inline mode (no space between the slashes and the parenthesis).

Then you'll be able to write:

$$
mse = (\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}
$$

$$ e=mc^2 $$
