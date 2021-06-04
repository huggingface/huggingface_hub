---
title: 🤗 Hub Inference API
---

<h1>🤗 Hub Inference API</h1>

For detailed usage documentation, please refer to [Accelerated Inference API Documentation](https://api-inference.huggingface.co/docs/python/html/index.html).

## Do you only support 🤗 Transformers?

No, the 🤗 Hub supports other libraries, and we're working on expanding this support!

Find all our supported libraries [here](https://github.com/huggingface/huggingface_hub/blob/main/interfaces/Libraries.ts)!

| Library               | Description                                                                   | Inference API | Widgets | Download from Hub | Push to Hub |
|-----------------------|-------------------------------------------------------------------------------|---------------|-------:|-------------------|-------------|
| [Adapter Transformers](https://github.com/Adapter-Hub/adapter-transformers)  | Extends 🤗Transformers with Adapters.                                          |       ❌       | ❌      |         ✅         |      ❌      |
| [Asteroid](https://github.com/asteroid-team/asteroid)              | Pytorch-based audio source separation toolkit                                 |       ✅       | ❌      |         ✅         |      ❌      |
| [ESPnet](https://github.com/espnet/espnet)                | End-to-end speech processing toolkit (e.g. TTS)                               |       ✅       | ✅      |         ✅         |      ❌      |
| [Flair](https://github.com/flairNLP/flair)                 | Very simple framework for state-of-the-art NLP. |       ✅       |    ✅   |         ✅         |      ❌      |
| [Pyannote](https://github.com/pyannote/pyannote-audio)              | Neural building blocks for speaker diarization.                               |       ❌       |    ❌   |         ✅         |      ❌      |
| [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) | Compute dense vector representations for sentences, paragraphs, and images.   |       ✅       |    ✅   |         ✅         |      ✅      |
| [spaCy](https://github.com/explosion/spaCy)                 | Advanced Natural Language Processing in Python and Cython.                    |       ✅       |    ✅   |         ✅         |      ❌      |
| [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)         | Real-time state-of-the-art speech synthesis architectures.                    |       ❌       |    ❌   |         ✅         |      ❌      |
| [Timm](https://github.com/rwightman/pytorch-image-models)                  | Collection of image models, scripts, pretrained weights, etc.                 |       ❌       |    ❌   |         ✅         |      ❌      |
| [🤗 Transformers](https://github.com/huggingface/transformers)         | State-of-the-art Natural Language Processing for Pytorch, TensorFlow, and JAX |       ✅       |    ✅   |         ✅         |      ✅      |

Would you like to see your library here? Head to [huggingface_hub](https://github.com/huggingface/huggingface_hub)!

## How can I add a new library to the Inference API?

You can find detailed instructions in [huggingface_hub](https://github.com/huggingface/huggingface_hub).

## What technology do you use to power the inference API?

For 🤗 Transformers models, the API is built on top of our [Pipelines](https://huggingface.co/transformers/main_classes/pipelines.html) feature.

On top of `Pipelines` and depending on the model type, we build a number of production optimizations like:
- compiling models to optimized intermediary representations (e.g. [ONNX](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)),
- maintaining a Least Recently Used cache ensuring that the most popular models are always loaded,
- scaling the underlying compute infrastructure on the fly depending on the load constraints.


## How can I turn off the inference API for my model?

Specify `inference: false` in your model card's metadata.


## Can I send large volumes of requests? Can I get accelerated APIs?

If you are interested in accelerated inference and/or higher volumes of requests and/or a SLA, please contact us at `api-enterprise at huggingface.co`.

## How can I see my usage?

You can head to the [Inference API dashboard](https://api-inference.huggingface.co/dashboard/). Learn more about it in the [Inference API documentation](https://api-inference.huggingface.co/docs/python/html/usage.html#api-usage-dashboard). 
