---
title: Model Repos docs
---

<h1>Model Repos docs</h1>

## What are model cards and why are they useful?

The model card are markdown files that accompany the models and provide very useful information. They are extremely important for discoverability, reproducibility and sharing! They are the `README.md` file in any repo.

## When sharing a model, what should I add to my model card?

The model card should describe:
- the model
- its intended uses & potential limitations, including bias and ethical considerations as detailed in [Mitchell, 2018](https://arxiv.org/abs/1810.03993)
- the training params and experimental info – you can embed or link to an experiment tracking platform for reference
- which datasets did you train on, and your eval results

If needed you can find a template [here](https://github.com/huggingface/model_card).

## How are model tags determined?

The model cards have a YAML section that specify metadata. Here is an example
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

Yes! You can specify the framework in the model card metadata section:

```
tags:
- flair
```

Find all our supported libraries [here](https://github.com/huggingface/huggingface_hub/blob/main/interfaces/Libraries.ts)!

## How can I link a model to a dataset?

You can specify the dataset in the description:

```
datasets:
- wmt19
```


## Can I access models programatically?

You can use the [huggingface_hub](https://github.com/huggingface/huggingface_hub) library to create, delete and update repos. You can also use it to download files from repos and integrate it to your own library! For example, you can easily load a Scikit learn model with few lines.

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
mse = (\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}
$$

$$ e=mc^2 $$

