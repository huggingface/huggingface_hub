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
* License identifiers are the keywords listed in the right column of [this table](#list-of-license-identifiers).
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

## List of license identifiers

Fullname | License identifier (to use in model card)
--- | ---
Academic Free License v3.0	| `afl-3.0`
Apache license 2.0	| `apache-2.0`
Artistic license 2.0	| `artistic-2.0`
Boost Software License 1.0	| `bsl-1.0`
BSD 2-clause "Simplified" license	| `bsd-2-clause`
BSD 3-clause "New" or "Revised" license	| `bsd-3-clause`
BSD 3-clause Clear license	| `bsd-3-clause-clear`
Creative Commons license family	| `cc`
Creative Commons Zero v1.0 Universal	| `cc0-1.0`
Creative Commons Attribution 4.0	| `cc-by-4.0`
Creative Commons Attribution Share Alike 4.0	| `cc-by-sa-4.0`
Do What The F*ck You Want To Public License	| `wtfpl`
Educational Community License v2.0	| `ecl-2.0`
Eclipse Public License 1.0	| `epl-1.0`
Eclipse Public License 2.0	| `epl-2.0`
European Union Public License 1.1	| `eupl-1.1`
GNU Affero General Public License v3.0	| `agpl-3.0`
GNU General Public License family	| `gpl`
GNU General Public License v2.0	| `gpl-2.0`
GNU General Public License v3.0	| `gpl-3.0`
GNU Lesser General Public License family	| `lgpl`
GNU Lesser General Public License v2.1	| `lgpl-2.1`
GNU Lesser General Public License v3.0	| `lgpl-3.0`
ISC	| `isc`
LaTeX Project Public License v1.3c	| `lppl-1.3c`
Microsoft Public License	| `ms-pl`
MIT	| `mit`
Mozilla Public License 2.0	| `mpl-2.0`
Open Software License 3.0	| `osl-3.0`
PostgreSQL License	| `postgresql`
SIL Open Font License 1.1	| `ofl-1.1`
University of Illinois/NCSA Open Source License	| `ncsa`
The Unlicense	| `unlicense`
zLib License	| `zlib`
