---
title: Hugging Face Hub docs
---

<h1>Hugging Face Hub documentation</h1>


## What's the Hugging Face Hub?

We are helping the community work together towards the goal of advancing Artificial Intelligence 🔥.

Not one company, even the Tech Titans, will be able to “solve AI” by themselves – the only way we'll achieve this is by sharing knowledge and resources. On the Hugging Face Hub we are building the largest collection of models, datasets and metrics in order to democratize and advance AI for everyone 🚀. The Hugging Face Hub works as a central place where anyone can share and explore models and datasets.

## What's a repository?

The Hugging Face Hub hosts Git-based repositories which are storage spaces that can contain all your files 💾.

These repositories have multiple advantages over other hosting solutions:

* versioning
* commit history and diffs
* branches

On top of that, Hugging Face Hub repositories have many other advantages:

* Repos provide useful [metadata](/docs/hub/model-repos#model-card-metadata) about their tasks, languages, metrics, etc.
* Anyone can play with the model directly in the browser!
* Training metrics charts are displayed if the repository contains [TensorBoard traces](https://huggingface.co/models?filter=tensorboard).
* An API is provided to use the models in production settings.
* [Over 10 frameworks](/docs/hub/libraries) such as 🤗 Transformers, Asteroid and ESPnet support using models from the Hugging Face Hub. 


## What's a widget?

Many repos have a widget that allows anyone to do inference directly on the browser!

Here are some examples:
* [Named Entity Recognition](https://huggingface.co/spacy/en_core_web_sm?text=My+name+is+Sarah+and+I+live+in+London) using [spaCy](https://spacy.io/).
* [Image Classification](https://huggingface.co/google/vit-base-patch16-224) using [🤗 Transformers](https://github.com/huggingface/transformers)
* [Text to Speech](https://huggingface.co/julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train) using [ESPnet](https://github.com/espnet/espnet).
* [Sentence Similarity](https://huggingface.co/osanseviero/full-sentence-distillroberta3) using [Sentence Transformers](https://github.com/UKPLab/sentence-transformers).

You can try out all the widgets [here](https://huggingface-widgets.netlify.app/).

## What's the Inference API?

The Inference API allows you to send HTTP requests to models in the Hugging Face Hub. The Inference API is 2x to 10x faster than the widgets! ⚡⚡


## How can I explore the Hugging Face Hub?

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/XvSGPZFEjDY" title="Model Hub Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## How is a model's type of inference API and widget determined?

To determine which pipeline and widget to display (`text-classification`, `token-classification`, `translation`, etc.), we analyze information in the repo such as the metadata provided in the model card and configuration files. This information is mapped to a single `pipeline_tag`. We choose to expose **only one** widget per model for simplicity.

For most use cases, the model type is determined from the tags. For example, if there is `tag: text-classification` in the metadata, the inferred `pipeline_tag` will be `text-classification`.

For `🤗 Transformers` however, the model type is determined automatically from `config.json`. The architecture can be used to determine the type: for example, `AutoModelForTokenClassification` corresponds to `token-classification`. If you're really interested in this, you can see pseudo-code in [this gist](https://gist.github.com/julien-c/857ba86a6c6a895ecd90e7f7cab48046).

You can always manually override your pipeline type with pipeline_tag: xxx in your model card metadata.


## What are all the possible task/widget types?

You can find all the supported tasks [here](https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Types.ts).

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

You have access to the repos as with any other Git-based repository! You can even upload very large files. 

Here is a video to learn more about it from our [course](http://hf.co/course)!

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/rkCly_cbMBk" title="Managing a repo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## What's the origin of 🤗 name? 

🤗
