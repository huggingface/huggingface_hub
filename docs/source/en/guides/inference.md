<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Run Inference on servers

Inference is the process of using a trained model to make predictions on new data. As this process can be compute-intensive,
running on a dedicated server can be an interesting option. The `huggingface_hub` library provides an easy way to call a
service that runs inference for hosted models. There are several services you can connect to:
- [Inference API](https://huggingface.co/docs/api-inference/index): a service that allows you to run accelerated inference
on Hugging Face's infrastructure for free. This service is a fast way to get started, test different models, and
prototype AI products.
- [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index): a product to easily deploy models to production.
Inference is run by Hugging Face in a dedicated, fully managed infrastructure on a cloud provider of your choice.

These services can be called with the [`InferenceClient`] object. It acts as a replacement for the legacy
[`InferenceApi`] client, adding specific support for tasks and handling inference on both
[Inference API](https://huggingface.co/docs/api-inference/index) and [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index).
Learn how to migrate to the new client in the [Legacy InferenceAPI client](#legacy-inferenceapi-client) section.

<Tip>

[`InferenceClient`] is a Python client making HTTP calls to our APIs. If you want to make the HTTP calls directly using
your preferred tool (curl, postman,...), please refer to the [Inference API](https://huggingface.co/docs/api-inference/index)
or to the [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) documentation pages.

For web development, a [JS client](https://huggingface.co/docs/huggingface.js/inference/README) has been released.
If you are interested in game development, you might have a look at our [C# project](https://github.com/huggingface/unity-api).

</Tip>

## Getting started

Let's get started with a text-to-image task:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> image = client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")  # 'image' is a PIL.Image object
```

In the example above, we initialized an [`InferenceClient`] with the default parameters. The only thing you need to know is the [task](#supported-tasks) you want to perform. By default, the client will connect to the Inference API and select a model to complete the task. In our example, we generated an image from a text prompt. The returned value is a `PIL.Image` object that can be saved to a file. For more details, check out the [`~InferenceClient.text_to_image`] documentation.

Let's now see an example using the [~`InferenceClient.chat_completion`] API. This task uses an LLM to generate a response from a list of messages:

```python
>>> from huggingface_hub import InferenceClient
>>> messages = [{"role": "user", "content": "What is the capital of France?"}]
>>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
>>> client.chat_completion(messages, max_tokens=100)
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason='eos_token',
            index=0,
            message=ChatCompletionOutputMessage(
                role='assistant',
                content='The capital of France is Paris.',
                name=None,
                tool_calls=None
            ),
            logprobs=None
        )
    ],
    created=1719907176,
    id='',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    object='text_completion',
    system_fingerprint='2.0.4-sha-f426a33',
    usage=ChatCompletionOutputUsage(
        completion_tokens=8,
        prompt_tokens=17,
        total_tokens=25
    )
)
```

In this example, we specified which model we want to use (`"meta-llama/Meta-Llama-3-8B-Instruct"`). You can find a list of compatible models [on this page](https://huggingface.co/models?other=conversational&sort=likes). We then gave a list of messages to complete (here, a single question) and passed an additional parameter to API (`max_token=100`). The output is a `ChatCompletionOutput` object that follows the OpenAI specification. The generated content can be access with `output.choices[0].message.content`. For more details, check out the [`~InferenceClient.chat_completion`] documentation.


<Tip warning={true}>

The API is designed to be simple. Not all parameters and options are available or described for the end user. Check out
[this page](https://huggingface.co/docs/api-inference/detailed_parameters) if you are interested in learning more about
all the parameters available for each task.

</Tip>

### Using a specific model

What if you want to use a specific model? You can specify it either as a parameter or directly at an instance level:

```python
>>> from huggingface_hub import InferenceClient
# Initialize client for a specific model
>>> client = InferenceClient(model="prompthero/openjourney-v4")
>>> client.text_to_image(...)
# Or use a generic client but pass your model as an argument
>>> client = InferenceClient()
>>> client.text_to_image(..., model="prompthero/openjourney-v4")
```

<Tip>

There are more than 200k models on the Hugging Face Hub! Each task in the [`InferenceClient`] comes with a recommended
model. Be aware that the HF recommendation can change over time without prior notice. Therefore it is best to explicitly
set a model once you are decided. Also, in most cases you'll be interested in finding a model specific to _your_ needs.
Visit the [Models](https://huggingface.co/models) page on the Hub to explore your possibilities.

</Tip>

### Using a specific URL

The examples we saw above use the Serverless Inference API. This proves to be very useful for prototyping
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

### Authentication

Calls made with the [`InferenceClient`] can be authenticated using a [User Access Token](https://huggingface.co/docs/hub/security-tokens).
By default, it will use the token saved on your machine if you are logged in (check out
[how to authenticate](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)). If you are not logged in, you can pass
your token as an instance parameter:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(token="hf_***")
```

<Tip>

Authentication is NOT mandatory when using the Inference API. However, authenticated users get a higher free-tier to
play with the service. Token is also mandatory if you want to run inference on your private models or on private
endpoints.

</Tip>

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

And that's it! The only required changes are to replace `from openai import OpenAI` by `from huggingface_hub import InferenceClient` and `client = OpenAI(...)` by `client = InferenceClient(...)`. You can chose any LLM model from the Hugging Face Hub by passing its model id as `model` parameter. [Here is a list](https://huggingface.co/models?pipeline_tag=text-generation&other=conversational,text-generation-inference&sort=trending) of supported models. For authentication, you should pass a valid [User Access Token](https://huggingface.co/settings/tokens) as `api_key` or authenticate using `huggingface_hub` (see the [authentication guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)).

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
1. [`InferenceClient`] is configured for Hugging Face services. You don't need to provide a `base_url` to run models on the serverless Inference API. You also don't need to provide a `token` or `api_key` if you machine is already correctly logged in.
2. [`InferenceClient`] is tailored for both Text-Generation-Inference (TGI) and `transformers` frameworks, meaning you are assured it will always be on-par with the latest updates.
3. [`InferenceClient`] is integrated with our Inference Endpoints service, making it easier to launch an Inference Endpoint, check its status and run inference on it. Check out the [Inference Endpoints](./inference_endpoints.md) guide for more details.

<Tip>

`InferenceClient.chat.completions.create` is simply an alias for `InferenceClient.chat_completion`. Check out the package reference of [`~InferenceClient.chat_completion`] for more details. `base_url` and `api_key` parameters when instantiating the client are also aliases for `model` and `token`. These aliases have been defined to reduce friction when switching from `OpenAI` to `InferenceClient`.

</Tip>

## Supported tasks

[`InferenceClient`]'s goal is to provide the easiest interface to run inference on Hugging Face models. It
has a simple API that supports the most common tasks. Here is a list of the currently supported tasks:

| Domain | Task                           | Supported    | Documentation                             |
|--------|--------------------------------|--------------|------------------------------------|
| Audio | [Audio Classification](https://huggingface.co/tasks/audio-classification)           | ✅ | [`~InferenceClient.audio_classification`] |
| Audio | [Audio-to-Audio](https://huggingface.co/tasks/audio-to-audio)           | ✅ | [`~InferenceClient.audio_to_audio`] |
| | [Automatic Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition)   | ✅ | [`~InferenceClient.automatic_speech_recognition`] |
| | [Text-to-Speech](https://huggingface.co/tasks/text-to-speech)                 | ✅ | [`~InferenceClient.text_to_speech`] |
| Computer Vision | [Image Classification](https://huggingface.co/tasks/image-classification)           | ✅ | [`~InferenceClient.image_classification`] |
| | [Image Segmentation](https://huggingface.co/tasks/image-segmentation)             | ✅ | [`~InferenceClient.image_segmentation`] |
| | [Image-to-Image](https://huggingface.co/tasks/image-to-image)                 | ✅ | [`~InferenceClient.image_to_image`] |
| | [Image-to-Text](https://huggingface.co/tasks/image-to-text)                  | ✅ | [`~InferenceClient.image_to_text`] |
| | [Object Detection](https://huggingface.co/tasks/object-detection)            | ✅ | [`~InferenceClient.object_detection`] |
| | [Text-to-Image](https://huggingface.co/tasks/text-to-image)                  | ✅ | [`~InferenceClient.text_to_image`] |
| | [Zero-Shot-Image-Classification](https://huggingface.co/tasks/zero-shot-image-classification)                  | ✅ | [`~InferenceClient.zero_shot_image_classification`] |
| Multimodal | [Documentation Question Answering](https://huggingface.co/tasks/document-question-answering) | ✅ | [`~InferenceClient.document_question_answering`]
| | [Visual Question Answering](https://huggingface.co/tasks/visual-question-answering)      | ✅ | [`~InferenceClient.visual_question_answering`] |
| NLP | Conversational                 |   | *deprecated*, use Chat Completion |
| | [Chat Completion](https://huggingface.co/tasks/text-generation)             | ✅ | [`~InferenceClient.chat_completion`] |
| | [Feature Extraction](https://huggingface.co/tasks/feature-extraction)             | ✅ | [`~InferenceClient.feature_extraction`] |
| | [Fill Mask](https://huggingface.co/tasks/fill-mask)                      | ✅ | [`~InferenceClient.fill_mask`] |
| | [Question Answering](https://huggingface.co/tasks/question-answering)             | ✅ | [`~InferenceClient.question_answering`]
| | [Sentence Similarity](https://huggingface.co/tasks/sentence-similarity)            | ✅ | [`~InferenceClient.sentence_similarity`] |
| | [Summarization](https://huggingface.co/tasks/summarization)                  | ✅ | [`~InferenceClient.summarization`] |
| | [Table Question Answering](https://huggingface.co/tasks/table-question-answering)       | ✅ | [`~InferenceClient.table_question_answering`] |
| | [Text Classification](https://huggingface.co/tasks/text-classification)            | ✅ | [`~InferenceClient.text_classification`] |
| | [Text Generation](https://huggingface.co/tasks/text-generation)   | ✅ | [`~InferenceClient.text_generation`] |
| | [Token Classification](https://huggingface.co/tasks/token-classification)           | ✅ | [`~InferenceClient.token_classification`] |
| | [Translation](https://huggingface.co/tasks/translation)       | ✅ | [`~InferenceClient.translation`] |
| | [Zero Shot Classification](https://huggingface.co/tasks/zero-shot-classification)       | ✅ | [`~InferenceClient.zero_shot_classification`] |
| Tabular | [Tabular Classification](https://huggingface.co/tasks/tabular-classification)         | ✅ | [`~InferenceClient.tabular_classification`] |
| | [Tabular Regression](https://huggingface.co/tasks/tabular-regression)             | ✅ | [`~InferenceClient.tabular_regression`] |

<Tip>

Check out the [Tasks](https://huggingface.co/tasks) page to learn more about each task, how to use them, and the
most popular models for each task.

</Tip>

## Custom requests

However, it is not always possible to cover all use cases. For custom requests, the [`InferenceClient.post`] method
gives you the flexibility to send any request to the Inference API. For example, you can specify how to parse the inputs
and outputs. In the example below, the generated image is returned as raw bytes instead of parsing it as a `PIL Image`.
This can be helpful if you don't have `Pillow` installed in your setup and just care about the binary content of the
image. [`InferenceClient.post`] is also useful to handle tasks that are not yet officially supported.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> response = client.post(json={"inputs": "An astronaut riding a horse on the moon."}, model="stabilityai/stable-diffusion-2-1")
>>> response.content # raw bytes
b'...'
```

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
# Code must be run in a asyncio concurrent context.
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

## Advanced tips

In the above section, we saw the main aspects of [`InferenceClient`]. Let's dive into some more advanced tips.

### Timeout

When doing inference, there are two main causes for a timeout:
- The inference process takes a long time to complete.
- The model is not available, for example when Inference API is loading it for the first time.

[`InferenceClient`] has a global `timeout` parameter to handle those two aspects. By default, it is set to `None`,
meaning that the client will wait indefinitely for the inference to complete. If you want more control in your workflow,
you can set it to a specific value in seconds. If the timeout delay expires, an [`InferenceTimeoutError`] is raised.
You can catch it and handle it in your code:

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
sending it to the Inference API.

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```

## Legacy InferenceAPI client

[`InferenceClient`] acts as a replacement for the legacy [`InferenceApi`] client. It adds specific support for tasks and
handles inference on both [Inference API](https://huggingface.co/docs/api-inference/index) and [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index).

Here is a short guide to help you migrate from [`InferenceApi`] to [`InferenceClient`].

### Initialization

Change from

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="bert-base-uncased", token=API_TOKEN)
```

to

```python
>>> from huggingface_hub import InferenceClient
>>> inference = InferenceClient(model="bert-base-uncased", token=API_TOKEN)
```

### Run on a specific task

Change from

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="paraphrase-xlm-r-multilingual-v1", task="feature-extraction")
>>> inference(...)
```

to

```python
>>> from huggingface_hub import InferenceClient
>>> inference = InferenceClient()
>>> inference.feature_extraction(..., model="paraphrase-xlm-r-multilingual-v1")
```

<Tip>

This is the recommended way to adapt your code to [`InferenceClient`]. It lets you benefit from the task-specific
methods like `feature_extraction`.

</Tip>

### Run custom request

Change from

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="bert-base-uncased")
>>> inference(inputs="The goal of life is [MASK].")
[{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]
```

to

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> response = client.post(json={"inputs": "The goal of life is [MASK]."}, model="bert-base-uncased")
>>> response.json()
[{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]
```

### Run with parameters

Change from

```python
>>> from huggingface_hub import InferenceApi
>>> inference = InferenceApi(repo_id="typeform/distilbert-base-uncased-mnli")
>>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
>>> params = {"candidate_labels":["refund", "legal", "faq"]}
>>> inference(inputs, params)
{'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}
```

to

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
>>> params = {"candidate_labels":["refund", "legal", "faq"]}
>>> response = client.post(json={"inputs": inputs, "parameters": params}, model="typeform/distilbert-base-uncased-mnli")
>>> response.json()
{'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}
```
