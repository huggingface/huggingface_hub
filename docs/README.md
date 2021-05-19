---
title: Model Hub docs
---

<h1 class="no-top-margin">Model Hub documentation</h1>


## What's the Hugging Face model hub

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

License identifiers are those standardized by GitHub [here](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#searching-github-by-license-type).

Dataset and metric identifiers are those listed on https://huggingface.co/datasets and in the [`nlp`](https://github.com/huggingface/nlp) repository.

All the tags can then be used to filter the list of models on https://huggingface.co/models.


## How are model tags determined?

On top of each model page (see e.g. [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)) you'll see the model's tags – they help for discovery and condition which features are enabled on which model page.

- The weight files that compose the models condition the framework(s) like `pytorch`, `tf`, etc.
- The `"architectures"` field of the model's config.json file – which should be automatically filled if you save your model using `.save_pretrained()` – condition the type of pipeline used in the inference API, and the type of widget present on the model page
	- A simplified snapshot of the mapping code can be found in [this gist]().
- If your config.json file contains a `task_specific_params` subfield, its sub-keys will be added as `pipeline:` tags. See [`t5-base`](http://localhost:5564/t5-base) for example.
- Most other metadata from the metadata block are also added as extra tags, at the end of the list.


## How is a model's type of inference API and widget determined?

To determine which pipeline and widget to display (text-classification, token-classification, translation, etc.), we use a simple mapping from model tags to one particular `pipeline_tag` (we currently only expose *one* pipeline and widget on each model page, even for models that would support several).

We try to use the most specific pipeline for each model, see pseudo-code in [this gist]().


## Can I send large volumes of requests? Can I get accelerated APIs?

If you are interested in accelerated inference and/or higher volumes of requests and/or a SLA, please contact us at `team at huggingface.co`.


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


