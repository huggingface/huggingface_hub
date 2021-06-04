---
title: 🤗 Hub docs
---

<h1>🤗 Hub documentation</h1>


## What's the 🤗 Hub?

We are helping the community work together towards the goal of advancing Artificial Intelligence 🔥.

Not one company, even the Tech Titans, will be able to “solve AI” by itself – the only way we'll achieve this is by sharing knowledge and resources. On the 🤗 Hub we are building the largest collection of models, datasets and metrics in order to democratize and advance AI for everyone 🚀. The 🤗 Hub works as a central place where anyone can share and explore models and datasets.


## What's a repository?

The 🤗 Hub hosts Git-based repositories which are storage spaces that can contain all your files 💾.

These repositories have multiple advantages over other hosting solutions:

* versioning
* commit history and diffs
* branches

On top of that, 🤗 Hub repositories have many other advantages:

* Repos provide useful metadata about their tasks, languages, metrics, etc.
* Anyone can play with the model directly in the browser!
* An API is provided to use the models in production settings.
* Over 10 frameworks such as 🤗 Transformers, Asteroid and ESPnet support using models from the 🤗 Hub. 


## What's a widget?

Many repos have a widget that allows anyone to do inference directly on the browser! Here is an example in which the model determines what's the most likely word in the middle of a sentence.

![A screenshot of a widget for the fill-token task.](/docs/assets/widget.png)


## What's the Inference API?

The Inference API allows doing simple HTTP requests to models in the 🤗 Hub. The Inference API is 2x to 10x faster than the widgets! ⚡⚡


## How can I explore the 🤗 Hub?
**Add video from Lysandre here**

## How is a model's type of inference API and widget determined?

To determine which pipeline and widget to display (text-classification, token-classification, translation, etc.), we analyze information in the repo such as the metadata provided in the model card and configuration files. This information is mapped to a single `pipeline_tag`. At the moment, we expose **only one** widget per model. We try to use the most specific pipeline for each model, see pseudo-code in [this gist](https://gist.github.com/julien-c/857ba86a6c6a895ecd90e7f7cab48046).

You can always manually override your pipeline type with `pipeline_tag: xxx` in your model card metadata.


## What are all the possible pipeline/widget types?

You can find all the supported pipelines [here](https://github.com/huggingface/huggingface_hub/blob/main/interfaces/Types.ts).

Here are some with links to examples:

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
- `sentence-similarity`, for instance [`osanseviero/full-sentence-distillroberta2`](/osanseviero/full-sentence-distillroberta2)


## How can I load/push from/to the Hub?

You have access to the repos as with any other Git-based repository! You can even upload very large files. Read more about it [here](https://huggingface.co/welcome).

TODO: Add video.

## What's the origin of 🤗 name? 

🤗