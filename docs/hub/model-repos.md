---
title: Model Repos docs
---

<h1>Model Repos docs</h1>

## What are model cards and why are they useful?

The model cards are markdown files that accompany the models and provide very useful information. They are extremely important for discoverability, reproducibility and sharing! They are the `README.md` file in any repo.

## When sharing a model, what should I add to my model card?

The model card should describe:
- the model
- its intended uses & potential limitations, including bias and ethical considerations as detailed in [Mitchell, 2018](https://arxiv.org/abs/1810.03993)
- the training params and experimental info (you can embed or link to an experiment tracking platform for reference)
- which datasets did you train on and your eval results


## Model card metadata
<!-- Try not to change this header as we use the corresponding anchor link -->

The model cards have a YAML section that specify metadata. These are the fields

```yaml
---
language: 
  - "List of ISO 639-1 code for your language"
  - lang1
  - lang2
thumbnail: "url to a thumbnail used in social sharing"
tags:
- tag1
- tag2
license: "any valid license identifier"
datasets:
- dataset1
- dataset2
metrics:
- metric1
- metric2
---
```

You can find the detailed specification [here](https://github.com/huggingface/huggingface_hub/blame/main/modelcard.md).


Some useful information on them:
* All the tags can be used to filter the list of models on https://huggingface.co/models.
* License identifiers are those standardized by GitHub in the right column (keywords) [here](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/licensing-a-repository#searching-github-by-license-type).
* Dataset, metric, and language identifiers are those listed on the [Datasets](https://huggingface.co/datasets), [Metrics](https://huggingface.co/metrics) and [Languages](https://huggingface.co/languages) pages and in the [`datasets`](https://github.com/huggingface/datasets) repository.


Here is an example: 
```yaml
---
language:
- ru
- en
tags:
- translation
license: apache-2.0
datasets:
- wmt19
metrics:
- bleu
- sacrebleu
---
```

## How are model tags determined?

Each model page lists all the model's tags in the page header, below the model name.

Those are primarily computed from the model card metadata, except that we also add some of them automatically, as described in [How is a model's type of inference API and widget determined?](/docs/hub/main#how-is-a-models-type-of-inference-api-and-widget-determined).

## How can I control my model's widget's example inputs?

You can specify the widget input in the model card metadata section:

```yaml
widget:
- text: "Jens Peter Hansen kommer fra Danmark"
```

We try to provide example inputs for some languages and most widget types in [this DefaultWidget.ts file](https://github.com/huggingface/huggingface_hub/blob/master/widgets/src/lib/interfaces/DefaultWidget.ts). If we lack some examples, please open a PR updating this file to add them. Thanks!

## Can I specify which framework supports my model?

Yes!🔥 You can specify the framework in the model card metadata section:

```yaml
tags:
- flair
```

Find more about our supported libraries [here](/docs/hub/libraries)!

## How can I link a model to a dataset?

You can specify the dataset in the metadata:

```yaml
datasets:
- wmt19
```


## Can I access models programatically?

You can use the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library to create, delete, update and retrieve information from repos. You can also use it to download files from repos and integrate it to your own library! For example, you can easily load a Scikit learn model with few lines.

```py
from huggingface_hub import hf_hub_url, cached_download
import joblib

REPO_ID = "YOUR_REPO_ID"
FILENAME = "sklearn_model.joblib"

model = joblib.load(cached_download(
    hf_hub_url(REPO_ID, FILENAME)
))
```

## Can I write LaTeX in my model card?

Yes, we use the [KaTeX](https://katex.org/) math typesetting library to render math formulas server-side, before parsing the markdown.

You have to use the following delimiters:
- `$$ ... $$` for display mode
- `\\` `(` `...` `\\` `)` for inline mode (no space between the slashes and the parenthesis).

Then you'll be able to write:

$$
\LaTeX
$$

$$
\mathrm{MSE} = \left(\frac{1}{n}\right)\sum_{i=1}^{n}(y_{i} - x_{i})^{2}
$$

$$ E=mc^2 $$
