<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Run Inference on servers

Inference is the process of using a trained model to make predictions on new data. Because this process can be compute-intensive, running on a dedicated or external service can be an interesting option.
The `huggingface_hub` library provides a unified interface to run inference across multiple services for models hosted on the Hugging Face Hub:

1.  [Inference Providers](https://huggingface.co/docs/inference-providers/index): a streamlined, unified access to hundreds of machine learning models, powered by our serverless inference partners. This new approach builds on our previous Serverless Inference API, offering more models, improved performance, and greater reliability thanks to world-class providers. Refer to the [documentation](https://huggingface.co/docs/inference-providers/index#partners) for a list of supported providers.
2.  [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index): a product to easily deploy models to production. Inference is run by Hugging Face in a dedicated, fully managed infrastructure on a cloud provider of your choice.
3.  Local endpoints: you can also run inference with local inference servers like [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm), [LiteLLM](https://docs.litellm.ai/docs/simple_proxy), or [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) by connecting the client to these local endpoints.

> [!TIP]
> [`InferenceClient`] is a Python client making HTTP calls to our APIs. If you want to make the HTTP calls directly using
> your preferred tool (curl, postman,...), please refer to the [Inference Providers](https://huggingface.co/docs/inference-providers/index) documentation
> or to the [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) documentation pages.
>
> For web development, a [JS client](https://huggingface.co/docs/huggingface.js/inference/README) has been released.
> If you are interested in game development, you might have a look at our [C# project](https://github.com/huggingface/unity-api).

## Getting started

Let's get started with a text-to-image task:

```python
>>> from huggingface_hub import InferenceClient

# Example with an external provider (e.g. replicate)
>>> replicate_client = InferenceClient(
    provider="replicate",
    api_key="my_replicate_api_key",
)
>>> replicate_image = replicate_client.text_to_image(
    "A flying car crossing a futuristic cityscape.",
    model="black-forest-labs/FLUX.1-schnell",
)
>>> replicate_image.save("flying_car.png")

```

In the example above, we initialized an [`InferenceClient`] with a third-party provider, [Replicate](https://replicate.com/). When using a provider, you must specify the model you want to use. The model id must be the id of the model on the Hugging Face Hub, not the id of the model from the third-party provider.
In our example, we generated an image from a text prompt. The returned value is a `PIL.Image` object that can be saved to a file. For more details, check out the [`~InferenceClient.text_to_image`] documentation.

Let's now see an example using the [`~InferenceClient.chat_completion`] API. This task uses an LLM to generate a response from a list of messages:

```python
>>> from huggingface_hub import InferenceClient
>>> messages = [
    {
        "role": "user",
        "content": "What is the capital of France?",
    }
]
>>> client = InferenceClient(
    provider="together",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="my_together_api_key",
)
>>> client.chat_completion(messages, max_tokens=100)
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason="eos_token",
            index=0,
            message=ChatCompletionOutputMessage(
                role="assistant", content="The capital of France is Paris.", name=None, tool_calls=None
            ),
            logprobs=None,
        )
    ],
    created=1719907176,
    id="",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    object="text_completion",
    system_fingerprint="2.0.4-sha-f426a33",
    usage=ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=17, total_tokens=25),
)
```

In the example above, we used a third-party provider ([Together AI](https://www.together.ai/)) and specified which model we want to use (`"meta-llama/Meta-Llama-3-8B-Instruct"`). We then gave a list of messages to complete (here, a single question) and passed an additional parameter to the API (`max_token=100`). The output is a `ChatCompletionOutput` object that follows the OpenAI specification. The generated content can be accessed with `output.choices[0].message.content`. For more details, check out the [`~InferenceClient.chat_completion`] documentation.


> [!WARNING]
> The API is designed to be simple. Not all parameters and options are available or described for the end user. Check out
> [this page](https://huggingface.co/docs/api-inference/detailed_parameters) if you are interested in learning more about
> all the parameters available for each task.

### Using a specific provider

If you want to use a specific provider, you can specify it when initializing the client. The default value is "auto" which will select the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers. Refer to the [Supported providers and tasks](#supported-providers-and-tasks) section for a list of supported providers.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="replicate", api_key="my_replicate_api_key")
```

### Using a specific model

What if you want to use a specific model? You can specify it either as a parameter or directly at an instance level:

```python
>>> from huggingface_hub import InferenceClient
# Initialize client for a specific model
>>> client = InferenceClient(provider="together", model="meta-llama/Llama-3.1-8B-Instruct")
>>> client.text_to_image(...)
# Or use a generic client but pass your model as an argument
>>> client = InferenceClient(provider="together")
>>> client.text_to_image(..., model="meta-llama/Llama-3.1-8B-Instruct")
```

> [!TIP]
> When using the "hf-inference" provider, each task comes with a recommended model from the 1M+ models available on the Hub.
> However, this recommendation can change over time, so it's best to explicitly set a model once you've decided which one to use.
> For third-party providers, you must always specify a model that is compatible with that provider.
>
> Visit the [Models](https://huggingface.co/models?inference=warm) page on the Hub to explore models available through Inference Providers.

### Using Inference Endpoints

The examples we saw above use inference providers. While these prove to be very useful for prototyping
and testing things quickly. Once you're ready to deploy your model to production, you'll need to use a dedicated infrastructure.
That's where [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) comes into play. It allows you to deploy
any model and expose it as a private API. Once deployed, you'll get a URL that you can connect to using exactly the same
code as before, changing only the `model` parameter:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# or
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

Note that you cannot specify both a URL and a provider - they are mutually exclusive. URLs are used to connect directly to deployed endpoints.

### Using local endpoints

You can use [`InferenceClient`] to run chat completion with local inference servers (llama.cpp, vllm, litellm server, TGI, mlx, etc.) running on your own machine. The API should be OpenAI API-compatible.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="http://localhost:8080")

>>> response = client.chat.completions.create(
...     messages=[
...         {"role": "user", "content": "What is the capital of France?"}
...     ],
...     max_tokens=100
... )
>>> print(response.choices[0].message.content)
```

> [!TIP]
> Similarily to the OpenAI Python client, [`InferenceClient`] can be used to run Chat Completion inference with any OpenAI REST API-compatible endpoint.

### Authentication

Authentication can be done in two ways:

**Routed through Hugging Face** : Use Hugging Face as a proxy to access third-party providers. The calls will be routed through Hugging Face's infrastructure using our provider keys, and the usage will be billed directly to your Hugging Face account.

You can authenticate using a [User Access Token](https://huggingface.co/docs/hub/security-tokens). You can provide your Hugging Face token directly using the `api_key` parameter:

```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="hf_****"  # Your HF token
)
```

If you *don't* pass an `api_key`, the client will attempt to find and use a token stored locally on your machine. This typically happens if you've previously logged in. See the [Authentication Guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) for details on login.

```python
>>> client = InferenceClient(
    provider="replicate",
    token="hf_****"  # Your HF token
)
```

**Direct access to provider**: Use your own API key to interact directly with the provider's service:
```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="r8_****"  # Your Replicate API key
)
```

For more details, refer to the [Inference Providers pricing documentation](https://huggingface.co/docs/inference-providers/pricing#routed-requests-vs-direct-calls).

## Supported providers and tasks

[`InferenceClient`]'s goal is to provide the easiest interface to run inference on Hugging Face models, on any provider. It has a simple API that supports the most common tasks. Here is a table showing which providers support which tasks:

| Task                                                | Black Forest Labs | Cerebras | Clarifai | Cohere | fal-ai | Featherless AI | Fireworks AI | Groq | HF Inference | Hyperbolic | Nebius AI Studio | Novita AI | Replicate | Sambanova | Together |
| --------------------------------------------------- | ----------------- | -------- | -------- | ------ | ------ | -------------- | ------------ | ---- | ------------ | ---------- | ---------------- | --------- | --------- | --------- | -------- |
| [`~InferenceClient.audio_classification`]           | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.audio_to_audio`]                 | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.automatic_speech_recognition`]   | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.chat_completion`]                | ❌                 | ✅        | ✅        | ✅      | ❌      | ✅              | ✅            | ✅    | ✅            | ✅          | ✅                | ✅         | ❌         | ✅         | ✅        |
| [`~InferenceClient.document_question_answering`]    | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.feature_extraction`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ✅                | ❌         | ❌         | ✅         | ❌        |
| [`~InferenceClient.fill_mask`]                      | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.image_classification`]           | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.image_segmentation`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.image_to_image`]                 | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ✅         | ❌         | ❌        |
| [`~InferenceClient.image_to_video`]                 | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ❌            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.image_to_text`]                  | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.object_detection`]               | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.question_answering`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.sentence_similarity`]            | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.summarization`]                  | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.table_question_answering`]       | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.text_classification`]            | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.text_generation`]                | ❌                 | ❌        | ❌        | ❌      | ❌      | ✅              | ❌            | ❌    | ✅            | ✅          | ✅                | ✅         | ❌         | ❌         | ✅        |
| [`~InferenceClient.text_to_image`]                  | ✅                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ✅            | ✅          | ✅                | ❌         | ✅         | ❌         | ✅        |
| [`~InferenceClient.text_to_speech`]                 | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ✅         | ❌         | ❌        |
| [`~InferenceClient.text_to_video`]                  | ❌                 | ❌        | ❌        | ❌      | ✅      | ❌              | ❌            | ❌    | ❌            | ❌          | ❌                | ❌         | ✅         | ❌         | ❌        |
| [`~InferenceClient.tabular_classification`]         | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.tabular_regression`]             | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.token_classification`]           | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.translation`]                    | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.visual_question_answering`]      | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.zero_shot_image_classification`] | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |
| [`~InferenceClient.zero_shot_classification`]       | ❌                 | ❌        | ❌        | ❌      | ❌      | ❌              | ❌            | ❌    | ✅            | ❌          | ❌                | ❌         | ❌         | ❌         | ❌        |

> [!TIP]
> Check out the [Tasks](https://huggingface.co/tasks) page to learn more about each task.

## OpenAI compatibility

The `chat_completion` task follows [OpenAI's Python client](https://github.com/openai/openai-python) syntax. What does it mean for you? It means that if you are used to play with `OpenAI`'s APIs you will be able to switch to `huggingface_hub.InferenceClient` to work with open-source models by updating just 2 line of code!

```diff
- from openai import OpenAI
+ from huggingface_hub import InferenceClient

- client = OpenAI(
+ client = InferenceClient(
    base_url=...,
    api_key=...,
)


output = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```

And that's it! The only required changes are to replace `from openai import OpenAI` by `from huggingface_hub import InferenceClient` and `client = OpenAI(...)` by `client = InferenceClient(...)`. You can choose any LLM model from the Hugging Face Hub by passing its model id as `model` parameter. [Here is a list](https://huggingface.co/models?pipeline_tag=text-generation&other=conversational,text-generation-inference&sort=trending) of supported models. For authentication, you should pass a valid [User Access Token](https://huggingface.co/settings/tokens) as `api_key` or authenticate using `huggingface_hub` (see the [authentication guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)).

All input parameters and output format are strictly the same. In particular, you can pass `stream=True` to receive tokens as they are generated. You can also use the [`AsyncInferenceClient`] to run inference using `asyncio`:

```diff
import asyncio
- from openai import AsyncOpenAI
+ from huggingface_hub import AsyncInferenceClient

- client = AsyncOpenAI()
+ client = AsyncInferenceClient()

async def main():
    stream = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

asyncio.run(main())
```

You might wonder why using [`InferenceClient`] instead of OpenAI's client? There are a few reasons for that:
1. [`InferenceClient`] is configured for Hugging Face services. You don't need to provide a `base_url` to run models with Inference Providers. You also don't need to provide a `token` or `api_key` if your machine is already correctly logged in.
2. [`InferenceClient`] is tailored for both Text-Generation-Inference (TGI) and `transformers` frameworks, meaning you are assured it will always be on-par with the latest updates.
3. [`InferenceClient`] is integrated with our Inference Endpoints service, making it easier to launch an Inference Endpoint, check its status and run inference on it. Check out the [Inference Endpoints](./inference_endpoints.md) guide for more details.

> [!TIP]
> `InferenceClient.chat.completions.create` is simply an alias for `InferenceClient.chat_completion`. Check out the package reference of [`~InferenceClient.chat_completion`] for more details. `base_url` and `api_key` parameters when instantiating the client are also aliases for `model` and `token`. These aliases have been defined to reduce friction when switching from `OpenAI` to `InferenceClient`.

## Function Calling

Function calling allows LLMs to interact with external tools, such as defined functions or APIs. This enables users to easily build applications tailored to specific use cases and real-world tasks.
`InferenceClient` implements the same tool calling interface as the OpenAI Chat Completions API. Here is a simple example of tool calling using [Nebius](https://nebius.com/) as the inference provider:

```python
from huggingface_hub import InferenceClient

tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Paris, France"
                        }
                    },
                    "required": ["location"],
                },
            }
        }
]

client = InferenceClient(provider="nebius")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
    {
        "role": "user",
        "content": "What's the weather like the next 3 days in London, UK?"
    }
    ],
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.tool_calls[0].function.arguments)

```

> [!TIP]
> Please refer to the providers' documentation to verify which models are supported by them for Function/Tool Calling.

## Structured Outputs & JSON Mode

InferenceClient supports JSON mode for syntactically valid JSON responses and Structured Outputs for schema-enforced responses. JSON mode provides machine-readable data without strict structure, while Structured Outputs guarantee both valid JSON and adherence to a predefined schema for reliable downstream processing.

We follow the OpenAI API specs for both JSON mode and Structured Outputs. You can enable them via the `response_format` argument. Here is an example of Structured Outputs using [Cerebras](https://www.cerebras.ai/) as the inference provider:

```python
from huggingface_hub import InferenceClient

json_schema = {
    "name": "book",
    "schema": {
        "properties": {
            "name": {
                "title": "Name",
                "type": "string",
            },
            "authors": {
                "items": {"type": "string"},
                "title": "Authors",
                "type": "array",
            },
        },
        "required": ["name", "authors"],
        "title": "Book",
        "type": "object",
    },
    "strict": True,
}

client = InferenceClient(provider="cerebras")


completion = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[
        {"role": "system", "content": "Extract the books information."},
        {"role": "user", "content": "I recently read 'The Great Gatsby' by F. Scott Fitzgerald."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": json_schema,
    },
)

print(completion.choices[0].message)
```
> [!TIP]
> Please refer to the providers' documentation to verify which models are supported by them for Structured Outputs and JSON Mode.

## Async client

An async version of the client is also provided, based on `asyncio` and `aiohttp`. You can either install `aiohttp`
directly or use the `[inference]` extra:

```sh
pip install --upgrade huggingface_hub[inference]
# or
# pip install aiohttp
```

After installation all async API endpoints are available via [`AsyncInferenceClient`]. Its initialization and APIs are
strictly the same as the sync-only version.

```py
# Code must be run in an asyncio concurrent context.
# $ python -m asyncio
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> async for token in await client.text_generation("The Huggingface Hub is", stream=True):
...     print(token, end="")
 a platform for sharing and discussing ML-related content.
```

For more information about the `asyncio` module, please refer to the [official documentation](https://docs.python.org/3/library/asyncio.html).

## MCP Client

The `huggingface_hub` library now includes an experimental [`MCPClient`], designed to empower Large Language Models (LLMs) with the ability to interact with external Tools via the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). This client extends an [`AsyncInferenceClient`] to seamlessly integrate Tool usage.

The [`MCPClient`] connects to MCP servers (either local `stdio` scripts or remote `http`/`sse` services) that expose tools. It feeds these tools to an LLM (via [`AsyncInferenceClient`]). If the LLM decides to use a tool, [`MCPClient`] manages the execution request to the MCP server and relays the Tool's output back to the LLM, often streaming results in real-time.

In the following example, we use [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model via [Nebius](https://nebius.com/) inference provider. We then add a remote MCP server, in this case, an SSE server which made the Flux image generation tool available to the LLM.

```python
import os

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient


async def main():
    async with MCPClient(
        provider="nebius",
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=os.environ["HF_TOKEN"],
    ) as client:
        await client.add_mcp_server(type="sse", url="https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse")

        messages = [
            {
                "role": "user",
                "content": "Generate a picture of a cat on the moon",
            }
        ]

        async for chunk in client.process_single_turn_with_tools(messages):
            # Log messages
            if isinstance(chunk, ChatCompletionStreamOutput):
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="")

            # Or tool calls
            elif isinstance(chunk, ChatCompletionInputMessage):
                print(
                    f"\nCalled tool '{chunk.name}'. Result: '{chunk.content if len(chunk.content) < 1000 else chunk.content[:1000] + '...'}'"
                )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```


For even simpler development, we offer a higher-level [`Agent`] class. This 'Tiny Agent' simplifies creating conversational Agents by managing the chat loop and state, essentially acting as a wrapper around [`MCPClient`]. It's designed to be a simple while loop built right on top of an [`MCPClient`]. You can run these Agents directly from the command line:


```bash
# install latest version of huggingface_hub with the mcp extra
pip install -U huggingface_hub[mcp]
# Run an agent that uses the Flux image generation tool
tiny-agents run julien-c/flux-schnell-generator

```

When launched, the Agent will load, list the Tools it has discovered from its connected MCP servers, and then it's ready for your prompts!

## Advanced tips

In the above section, we saw the main aspects of [`InferenceClient`]. Let's dive into some more advanced tips.

### Billing

As an HF user, you get monthly credits to run inference through various providers on the Hub. The amount of credits you get depends on your type of account (Free or PRO or Enterprise Hub). You get charged for every inference request, depending on the provider's pricing table. By default, the requests are billed to your personal account. However, it is possible to set the billing so that requests are charged to an organization you are part of by simply passing `bill_to="<your_org_name>"` to `InferenceClient`. For this to work, your organization must be subscribed to Enterprise Hub. For more details about billing, check out [this guide](https://huggingface.co/docs/api-inference/pricing#features-using-inference-providers).

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="fal-ai", bill_to="openai")
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```

Note that it is NOT possible to charge another user or an organization you are not part of. If you want to grant someone else some credits, you must create a joint organization with them.


### Timeout

Inference calls can take a significant amount of time. By default, [`InferenceClient`] will wait "indefinitely" until the inference complete. If you want more control in your workflow, you can set the `timeout` parameter to a specific value in seconds. If the timeout delay expires, an [`InferenceTimeoutError`] is raised, which you can catch in your code:

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### Binary inputs

Some tasks require binary inputs, for example, when dealing with images or audio files. In this case, [`InferenceClient`]
tries to be as permissive as possible and accept different types:
- raw `bytes`
- a file-like object, opened as binary (`with open("audio.flac", "rb") as f: ...`)
- a path (`str` or `Path`) pointing to a local file
- a URL (`str`) pointing to a remote file (e.g. `https://...`). In this case, the file will be downloaded locally before
being sent to the API.

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```
