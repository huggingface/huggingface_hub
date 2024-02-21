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

## Return types

For most tasks, the return value has a built-in type (string, list, image...). Here is a list for the more complex types.

### ConversationalOutputConversation

[[autodoc]] huggingface_hub.inference._types.ConversationalOutputConversation

### ConversationalOutput

[[autodoc]] huggingface_hub.inference._types.ConversationalOutput

### ImageSegmentationOutput

[[autodoc]] huggingface_hub.inference._types.ImageSegmentationOutput

### ModelStatus

[[autodoc]] huggingface_hub.inference._common.ModelStatus

### TokenClassificationOutput

[[autodoc]] huggingface_hub.inference._types.TokenClassificationOutput

### Text generation types

[`~InferenceClient.text_generation`] task has a greater support than other tasks in `InferenceClient`. In
particular, user inputs and server outputs are validated using [Pydantic](https://docs.pydantic.dev/latest/)
if this package is installed. Therefore, we recommend installing it (`pip install pydantic`)
for a better user experience.

You can find below the dataclasses used to validate data and in particular [`~huggingface_hub.inference._text_generation.TextGenerationParameters`] (input),
[`~huggingface_hub.inference._text_generation.TextGenerationResponse`] (output) and
[`~huggingface_hub.inference._text_generation.TextGenerationStreamResponse`] (streaming output).

[[autodoc]] huggingface_hub.inference._text_generation.TextGenerationParameters

[[autodoc]] huggingface_hub.inference._text_generation.TextGenerationResponse

[[autodoc]] huggingface_hub.inference._text_generation.TextGenerationStreamResponse

[[autodoc]] huggingface_hub.inference._text_generation.InputToken

[[autodoc]] huggingface_hub.inference._text_generation.Token

[[autodoc]] huggingface_hub.inference._text_generation.FinishReason

[[autodoc]] huggingface_hub.inference._text_generation.BestOfSequence

[[autodoc]] huggingface_hub.inference._text_generation.Details

[[autodoc]] huggingface_hub.inference._text_generation.StreamDetails

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
