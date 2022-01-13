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

## How can I control my model's widget example input?

You can specify the widget input in the model card metadata section:

```yaml
widget:
- text: "Jens Peter Hansen kommer fra Danmark"
```

You can provide more than one example input. In the examples dropdown menu of the widget, they will appear as `Example 1`, `Example 2`, etc. Optionally, you can supply `example_title` as well.

![/docs/assets/hub/widget_input_examples.gif](/docs/assets/hub/widget_input_examples.gif)

```yaml
widget:
- text: "Is this review positive or negative? Review: Best cast iron skillet you will every buy."
  example_title: "Sentiment analysis"
- text: "Barack Obama nominated Hilary Clinton as his secretary of state on Monday. He chose her because she had ..."
  example_title: "Coreference resolution"
- text: "On a shelf, there are five books: a gray book, a red book, a purple book, a blue book, and a black book ..."
  example_title: "Logic puzzles"
- text: "The two men running to become New York City's next mayor will face off in their first debate Wednesday night ..."
  example_title: "Reading comprehension"
```

Moreover, you can specify non-text example inputs in the model card metadata. Refer [here](https://github.com/huggingface/huggingface_hub/blob/main/docs/hub/input-examples.md) for a complete list of example input formats for all widget types. For vision & audio widget types, provide example inputs with `src` rather than `text`. 

For example, allow users to choose from two sample audio files for automatic speech recognition tasks by:

```yaml
widget:
- src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
  example_title: Librispeech sample 1
- src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
  example_title: Librispeech sample 2
```

We provide example inputs for some languages and most widget types in [the DefaultWidget.ts file](https://github.com/huggingface/huggingface_hub/blob/master/js/src/lib/interfaces/DefaultWidget.ts). If some examples are missing, we welcome PRs from the community to add them!

## How can I control my model's widget Inference API parameters?

Generally, the Inference API for a model uses the default pipeline settings associated with each task. But if you'd like to change the pipeline's default settings and specify additional inference parameters, you can configure the parameters directly through the model card metadata. Refer [here](https://api-inference.huggingface.co/docs/python/html/detailed_parameters.html#) for some of the most commonly used parameters associated with each task.

For example, if you want to specify an aggregation strategy for a NER task in the widget:

```yaml
inference:
  parameters:
    aggregation_strategy: "none"
```

Or if you'd like to change the temperature for a summarization task in the widget:

```yaml
inference:
  parameters:
    temperature: 0.7
```

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

## How can I fork or rebase a repository with LFS pointers?

When you want to fork or [rebase](https://git-scm.com/docs/git-rebase) a repository with [LFS](https://git-lfs.github.com/) files (all files over 20MB are stored as such), you cannot use the usual Git approach since you need to be careful to not break the LFS pointers. Forking can take time depending on your bandwidth, because you will have to fetch an re-upload all the LFS files in your fork.

For example, say you have an upstream repository, **upstream**, and you just created your own repository on the Hub which is **myfork** in this example.

1. Create a destination repository (e.g. **myfork**) in https://huggingface.co 

2. Clone your fork repository

```
git lfs clone https://huggingface.co/me/myfork.git
```

3. Fetch non LFS files

```
cd myfork
git lfs install --skip-smudge --local # affects only this clone
git remote add upstream https://huggingface.co/friend/upstream.git
git fetch upstream
```

4. Fetch large files. This can take some time depending on your download bandwidth

```
git lfs fetch --all upstream # this can take time depending on your download bandwidth
```

4.a. If you want to override completely the fork history (which should only have an initial commit), run:

```
git reset --hard upstream/main
```

4.b. If you want to rebase instead of overriding, run the following command and solve any conflicts

```
git rebase upstream/main
```

5. Prepare your LFS files to push

```
git lfs install --force --local # this reinstalls the LFS hooks
huggingface-cli lfs-enable-largefiles . # needed if some files are bigger than 5Gb
```

6. And finally push

```
git push --force origin main # this can take time depending on your upload bandwidth
```

Now you have your own fork or rebased repo in the Hub!

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
Creative Commons Attribution 3.0	| `cc-by-3.0`
Creative Commons Attribution 4.0	| `cc-by-4.0`
Creative Commons Attribution Share Alike 3.0	| `cc-by-sa-3.0`
Creative Commons Attribution Share Alike 4.0	| `cc-by-sa-4.0`
Creative Commons Attribution Non Commercial 3.0	|`cc-by-nc-3.0`
Creative Commons Attribution Non Commercial 4.0	|`cc-by-nc-4.0`
Creative Commons Attribution Non Commercial Share Alike  3.0| `cc-by-nc-sa-3.0`
Creative Commons Attribution Non Commercial Share Alike  4.0| `cc-by-nc-sa-4.0`
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
Open Data Commons Public Domain Dedication and License | `pddl`
Lesser General Public License For Linguistic Resources | `lgpl-lr`

## Why is it useful to calculate the carbon emissions of my model?

Training ML models is often energy intensive and can produce a substantial carbon footprint, as described by [Strubell et al.](https://arxiv.org/abs/1906.02243). It's therefore important to *track* and *report* the emissions of models to get a better idea of the environmental impacts of our field.


## What information should I include about the carbon footprint of my model?

If you can, include information about:
- where the model was trained (in terms of location)
- the hardware that was used -- e.g. GPU, TPU or CPU, and how many
- training type: pre-training or fine-tuning
- the estimated carbon footprint of the model, calculated in real-time with the [Code Carbon](https://github.com/mlco2/codecarbon) package or after the fact using the [ML CO2 Calculator](https://mlco2.github.io/impact/).

## Carbon footprint metadata

The data can be added to the model card metadata (README.md file). The structure of the metadata should be:

```yaml
---
co2_eq_emissions:
	  emissions: "in grams of CO2"
	  source: "source of the information, either directly from autonlp, code carbon or from a scientific article documenting the model"
	  training_type: "pretraining or fine-tuning"
	  geographical_location: "as granular as possible, for instance Quebec, Canada or Brooklyn, NY, USA"
	  hardware_used: "how much compute and what kind, e.g. 8 v100 GPUs"
---
```

## How is the carbon footprint of my model calculated? 🌎

By taking into account the computing hardware, location, usage and training time, it's possible to provide an estimate of how much CO<sub>2</sub> was produced by the model.

The math is actually pretty simple! ➕

First, you take the *carbon intensity* of the electric grid that is being used for the training -- this is how much CO<sub>2</sub> is produced by KwH of electricity used. This depends on the location where the hardware is located and the [energy mix](https://electricitymap.org/) used at that location -- whether it's renewable energy like solar 🌞, wind 🌬️ and hydro 💧, or non-renewable energy like coal ⚫ and natural gas 💨. The more renewable energy is used, the less carbon intensive it is!
 
Then, you take the power consumption of the GPU during training -- this is done using the `pynvml` library.

Finally, you multiply the power consumption and carbon intensity by the training time of the model, and you have an estimate of how much CO<sub>2</sub> was emitted.

Keep in mind that this isn't an exact number, because there are other factors that come into play -- like the energy that's used for data center heating and cooling -- which will increase carbon emissions. But this will already give you a good idea of the scale of CO<sub>2</sub> emissions that your model is producing!

To add **Carbon Emissions** metadata to your models:

1. If you are using **AutoNLP**, this is tracked for you 🔥
2. Otherwise, use a tracker like  Code Carbon in your training code, then specify `co2_eq_emissions.emissions: 1.2345` in your model card metadata, where `1.2345` is the emissions value in **grams**. 

To learn more about the carbon footprint of Transformers, check out the [video](https://www.youtube.com/watch?v=ftWlj4FBHTg), part of the Hugging Face Course!
