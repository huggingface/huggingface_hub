<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Inference

Inference is the process of using a trained model to make predictions on new data. As this process can be compute-intensive,
running on a dedicated server can be an interesting option. The `huggingface_hub` library provides an easy way to call a
service that runs inference for hosted models. There are several services you can connect to:
- [Inference API](https://huggingface.co/docs/api-inference/index): a service that allows you to run accelerated inference
on Hugging Face's infrastructure for free. This service is a fast way to get started, test different models, and
prototype AI products.
- [Inference Endpoints](https://huggingface.co/inference-endpoints): a product to easily deploy models to production.
Inference is run by Hugging Face in a dedicated, fully managed infrastructure on a cloud provider of your choice.

These services can be called with the [`InferenceClient`] object. Please refer to [this guide](../guides/inference)
for more information on how to use it.

## Inference Client

[[autodoc]] InferenceClient

## Async Inference Client

An async version of the client is also provided, based on `asyncio` and `aiohttp`.
To use it, you can either install `aiohttp` directly or use the `[inference]` extra:

```sh
pip install --upgrade huggingface_hub[inference]
# or
# pip install aiohttp
```

[[autodoc]] AsyncInferenceClient

## InferenceTimeoutError

[[autodoc]] InferenceTimeoutError

### ModelStatus

[[autodoc]] huggingface_hub.inference._common.ModelStatus

## InferenceAPI

[`InferenceAPI`] is the legacy way to call the Inference API. The interface is more simplistic and requires knowing
the input parameters and output format for each task. It also lacks the ability to connect to other services like
Inference Endpoints or AWS SageMaker. [`InferenceAPI`] will soon be deprecated so we recommend using [`InferenceClient`]
whenever possible. Check out [this guide](../guides/inference#legacy-inferenceapi-client) to learn how to switch from
[`InferenceAPI`] to [`InferenceClient`] in your scripts.

[[autodoc]] InferenceApi
    - __init__
    - __call__
    - all
