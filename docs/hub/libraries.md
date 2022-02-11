---
title: Hugging Face Hub Libraries Docs
---

<h1>Hugging Face Hub Libraries</h1>


## Do you only support 🤗 Transformers?

No, the Hub supports other libraries and we're working on expanding this support! We're happy to welcome to the Hub a set of Open Source libraries that are pushing Machine Learning forward.

The table below summarizes the supported libraries and how they are integrated. Find all our supported libraries [here](https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Libraries.ts)! 

| Library               | Description                                                                   | Inference API | Widgets | Download from Hub | Push to Hub |
|-----------------------|-------------------------------------------------------------------------------|---------------|-------:|-------------------|-------------|
| [🤗 Transformers](https://github.com/huggingface/transformers)         | State-of-the-art Natural Language Processing for Pytorch, TensorFlow, and JAX |       ✅       |    ✅   |         ✅         |      ✅      |
| [Adapter Transformers](https://github.com/Adapter-Hub/adapter-transformers)  | Extends 🤗Transformers with Adapters.                                          |       ❌       | ❌      |         ✅         |      ✅      |
| [AllenNLP](https://github.com/allenai/allennlp)              | An open-source NLP research library, built on PyTorch.                        |       ✅       |    ✅   |         ✅         |      ❌      |
| [Asteroid](https://github.com/asteroid-team/asteroid)              | Pytorch-based audio source separation toolkit                                 |       ✅       | ✅     |         ✅         |      ❌      |
| [ESPnet](https://github.com/espnet/espnet)                | End-to-end speech processing toolkit (e.g. TTS)                               |       ✅       | ✅      |         ✅         |      ❌      |
| [Flair](https://github.com/flairNLP/flair)                 | Very simple framework for state-of-the-art NLP. |       ✅       |    ✅   |         ✅         |      ❌      |
| [Pyannote](https://github.com/pyannote/pyannote-audio)              | Neural building blocks for speaker diarization.                               |       ❌       |    ❌   |         ✅         |      ❌      |
| [PyCTCDecode](https://github.com/kensho-technologies/pyctcdecode)                  | Language model supported CTC decoding for speech recognition                |       ❌       |    ❌   |         ✅         |      ❌      |
| [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) | Compute dense vector representations for sentences, paragraphs, and images.   |       ✅       |    ✅   |         ✅         |      ✅      |
| [spaCy](https://github.com/explosion/spaCy)                 | Advanced Natural Language Processing in Python and Cython.                    |       ✅       |    ✅   |         ✅         |      ✅      |
| [Speechbrain](https://speechbrain.github.io/)                 | A PyTorch Powered Speech Toolkit. |       ✅       |    ✅   |         ✅         |      ❌      |
| [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)         | Real-time state-of-the-art speech synthesis architectures.                    |       ❌       |    ❌   |         ✅         |      ❌      |
| [Timm](https://github.com/rwightman/pytorch-image-models)                  | Collection of image models, scripts, pretrained weights, etc.                 |       ❌       |    ❌   |         ✅         |      ❌      |


## How can I add a new library to the Inference API?

Read about it in [Adding a Library Guide](/docs/hub/adding-a-library).
