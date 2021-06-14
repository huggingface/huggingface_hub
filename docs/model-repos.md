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

If needed you can find the specification [here](https://raw.githubusercontent.com/huggingface/huggingface_hub/main/modelcard.md).

## How are model tags determined?

The model cards have a YAML section that specify metadata. These are the fields

```
---
language: "ISO 639-1 code for your language, or `multilingual`"
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

Some useful information on them:
* All the tags can be used to filter the list of models on https://huggingface.co/models.
* License identifiers are those standardized by GitHub in the right column (keywords) [here](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/licensing-a-repository#searching-github-by-license-type).
* Dataset, metric, and language identifiers are those listed on the [Datasets](https://huggingface.co/datasets), [Metrics](https://huggingface.co/metrics) and [Languages](https://huggingface.co/languages) pages and in the [`datasets`](https://github.com/huggingface/datasets) repository.


Here is an example: 
```
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

## How can I control my model's widget's example inputs?

You can specify the widget input in the model card metadata section:

```
widget:
- text: "Jens Peter Hansen kommer fra Danmark"
```

## Can I specify which framework supports my model?

Yes!🔥 You can specify the framework in the model card metadata section:

```
tags:
- flair
```

Find more about our supported libraries [here](/docs/libraries)!

## How can I link a model to a dataset?

You can specify the dataset in the metadata:

```
datasets:
- wmt19
```


## Can I access models programatically?

You can use the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library to create, delete, update and retrieve information from repos. You can also use it to download files from repos and integrate it to your own library! For example, you can easily load a Scikit learn model with few lines.

```
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
