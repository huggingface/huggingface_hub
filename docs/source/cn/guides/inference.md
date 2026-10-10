<!--⚠️ 请注意，该文件虽然是 Markdown 格式，但包含文档生成器的特定语法（类似于 MDX），在某些 Markdown 查看器中可能无法正确渲染。
-->

# 在服务器上运行推理

推理是使用训练好的模型对新数据进行预测的过程。由于这个过程可能需要大量计算资源，在专用服务或外部服务上运行推理是一个不错的选择。
`huggingface_hub` 库为 Hub 上托管的模型提供了统一接口，可通过多个服务运行推理：

1.  [Inference Providers](https://huggingface.co/docs/inference-providers/index)：由无服务器推理合作伙伴提供，统一访问数百个机器学习模型。这个新方案是在旧版 Serverless Inference API 的基础上构建的，提供更多模型、更高性能，并凭借一流的服务商带来更高可靠性。支持的服务商请参阅[文档](https://huggingface.co/docs/inference-providers/index#partners)中的列表。
2.  [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)：用于将模型轻松部署到生产环境的产品。Hugging Face 会在你选择的云服务商上，以专用且完全托管的基础设施运行推理。
3.  本地端点：你也可以连接到本地端点，使用 [llama.cpp](https://github.com/ggerganov/llama.cpp)、[Ollama](https://ollama.com/)、[vLLM](https://github.com/vllm-project/vllm)、[LiteLLM](https://docs.litellm.ai/docs/simple_proxy) 或 [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) 等本地推理服务器运行推理。

> [!TIP]
> [`InferenceClient`] 是一个通过 HTTP 调用我们 API 的 Python 客户端。如果你想使用自己偏好的工具（curl、postman 等）直接发起 HTTP 请求，请参阅 [Inference Providers](https://huggingface.co/docs/inference-providers/index) 文档，
> 或 [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) 文档页面。
>
> 我们也发布了用于 Web 开发的 [JS 客户端](https://huggingface.co/docs/huggingface.js/inference/README)。
> 如果你对游戏开发感兴趣，可以了解我们的 [C# 项目](https://github.com/huggingface/unity-api)。

## 入门

下面从文本生成图像任务开始：

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

上面的示例使用第三方服务商 [Replicate](https://replicate.com/) 初始化了 [`InferenceClient`]。使用服务商时，必须指定要使用的模型。模型 ID 必须是 Hugging Face Hub 上的模型 ID，而不是第三方服务商中的模型 ID。
在示例中，我们根据文本提示生成了一张图像。返回值是一个可以保存到文件的 `PIL.Image` 对象。更多详情请参阅 [`~InferenceClient.text_to_image`] 文档。

下面看一个使用 [`~InferenceClient.chat_completion`] API 的示例。该任务使用 LLM 根据消息列表生成响应：

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

上面的示例使用了第三方服务商 [Together AI](https://www.together.ai/)，并指定了要使用的模型（`"meta-llama/Meta-Llama-3-8B-Instruct"`）。随后，我们提供了要补全的消息列表（这里是一条问题），并向 API 传入了额外参数（`max_token=100`）。输出是一个遵循 OpenAI 规范的 `ChatCompletionOutput` 对象。生成的内容可以通过 `output.choices[0].message.content` 获取。更多详情请参阅 [`~InferenceClient.chat_completion`] 文档。


> [!WARNING]
> 该 API 的设计目标是简单易用，并不会向最终用户提供或描述所有参数和选项。如果你想详细了解每个任务可用的全部参数，请查看[此页面](https://huggingface.co/docs/api-inference/detailed_parameters)。

### 使用指定的服务商

如果想使用指定的服务商，可以在初始化客户端时指定。默认值为 `"auto"`，它会根据用户在 https://hf.co/settings/inference-providers 中设置的顺序，为模型选择可用服务商列表中的第一个。支持的服务商请参阅[支持的服务商和任务](#supported-providers-and-tasks)部分。

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="replicate", api_key="my_replicate_api_key")
```

### 使用指定的模型

如果想使用指定的模型，可以将模型作为参数传入，也可以直接在实例级别指定：

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
> 使用 `"hf-inference"` 服务商时，每个任务都会从 Hub 上超过 100 万个模型中提供一个推荐模型。
> 但是，这个推荐可能会随时间变化，因此确定模型后，最好显式设置模型。
> 对于第三方服务商，必须始终指定一个与该服务商兼容的模型。
>
> 访问 Hub 上的[模型](https://huggingface.co/models?inference=warm)页面，探索 Inference Providers 支持的模型。

### 使用 Inference Endpoints

上面的示例使用了推理服务商。它们非常适合快速原型开发和测试。准备将模型部署到生产环境后，则需要专用基础设施。
这时可以使用 [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)。它可以部署任意模型并将其暴露为私有 API。部署完成后，你会获得一个 URL，只需修改 `model` 参数，就能使用与之前完全相同的代码连接到它：

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# or
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

注意，不能同时指定 URL 和服务商，它们是互斥的。URL 用于直接连接到已部署的端点。

### 使用本地端点

你可以使用 [`InferenceClient`]，在本机运行的本地推理服务器（llama.cpp、vllm、litellm server、TGI、mlx 等）上执行聊天补全。API 应与 OpenAI API 兼容。

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
> 与 OpenAI Python 客户端类似，[`InferenceClient`] 可以使用任意兼容 OpenAI REST API 的端点运行聊天补全推理。

### 身份验证

身份验证有两种方式：

**通过 Hugging Face 路由**：使用 Hugging Face 作为访问第三方服务商的代理。请求会通过 Hugging Face 的基础设施，并使用我们的服务商密钥路由；相关用量会直接计费到你的 Hugging Face 账户。

你可以使用[用户访问令牌](https://huggingface.co/docs/hub/security-tokens)进行身份验证。可以通过 `api_key` 参数直接传入 Hugging Face 令牌：

```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="hf_****"  # Your HF token
)
```

如果不传入 `api_key`，客户端会尝试查找并使用本机保存的令牌。这通常发生在你之前登录过的情况下。有关登录的详情，请参阅[身份验证指南](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)。

```python
>>> client = InferenceClient(
    provider="replicate",
    token="hf_****"  # Your HF token
)
```

**直接访问服务商**：使用自己的 API 密钥直接与服务商交互：
```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="r8_****"  # Your Replicate API key
)
```

更多详情请参阅 [Inference Providers 计费文档](https://huggingface.co/docs/inference-providers/pricing#routed-requests-vs-direct-calls)。

## 支持的服务商和任务

[`InferenceClient`] 旨在为你提供最简单的接口，使你可以在任意服务商上运行 Hugging Face 模型。它提供了支持常见任务的简单 API。下表展示了各服务商支持的任务：

| 任务                                                | Baseten | Cerebras | Cohere | DeepInfra | fal-ai | Featherless AI | Fireworks AI | Groq | HF Inference | Novita AI | Nscale | OVHcloud AI Endpoints | Public AI | Replicate | Scaleway | Together | Wavespeed | Zai |
| --------------------------------------------------- | ------- | -------- | ------ | --------- | ------ | -------------- | ------------ | ---- | ------------ | --------- | ------ | --------------------- | --------- | --------- | -------- | -------- | --------- | --- |
| [`~InferenceClient.audio_classification`]           | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.audio_to_audio`]                 | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.automatic_speech_recognition`]   | ❌      | ❌        | ❌      | ✅         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ✅         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.chat_completion`]                | ✅      | ✅        | ✅      | ✅         | ❌      | ✅              | ✅            | ✅    | ✅            | ✅         | ✅      | ✅                     | ✅         | ❌         | ✅        | ✅        | ❌        | ✅   |
| [`~InferenceClient.document_question_answering`]    | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.feature_extraction`]             | ❌      | ❌        | ❌      | ✅         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ✅        | ✅        | ❌        | ❌   |
| [`~InferenceClient.fill_mask`]                      | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.image_classification`]           | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.image_segmentation`]             | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.image_to_image`]                 | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ✅         | ❌        | ✅        | ✅        | ❌   |
| [`~InferenceClient.image_to_video`]                 | ❌      | ❌        | ❌      | ❌         | ✅      | ❌            | ❌            | ❌    | ❌            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ✅        | ✅        | ❌   |
| [`~InferenceClient.image_to_text`]                  | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.object_detection`]               | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.question_answering`]             | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.sentence_similarity`]            | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ✅        | ✅        | ❌        | ❌   |
| [`~InferenceClient.summarization`]                  | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.table_question_answering`]       | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.text_classification`]            | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.text_generation`]                | ❌      | ❌        | ❌      | ✅         | ❌      | ✅              | ❌            | ❌    | ✅            | ✅         | ❌      | ❌                     | ❌         | ❌         | ❌        | ✅        | ❌        | ❌   |
| [`~InferenceClient.text_to_image`]                  | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ✅      | ❌                     | ❌         | ✅         | ❌        | ✅        | ✅        | ✅   |
| [`~InferenceClient.text_to_speech`]                 | ❌      | ❌        | ❌      | ✅         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ✅         | ❌        | ✅        | ❌        | ❌   |
| [`~InferenceClient.text_to_video`]                  | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ❌            | ✅         | ❌      | ❌                     | ❌         | ✅         | ❌        | ✅        | ✅        | ❌   |
| [`~InferenceClient.tabular_classification`]         | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.tabular_regression`]             | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.token_classification`]           | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.translation`]                    | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.visual_question_answering`]      | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.zero_shot_image_classification`] | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌         | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |
| [`~InferenceClient.zero_shot_classification`]       | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌        | ❌   |

> [!TIP]
> 查看 [Tasks](https://huggingface.co/tasks) 页面，了解更多任务信息。

## OpenAI 兼容性

`chat_completion` 任务遵循 [OpenAI Python 客户端](https://github.com/openai/openai-python)的语法。这意味着，如果你熟悉使用 `OpenAI` 的 API，只需修改两行代码，就可以切换到 `huggingface_hub.InferenceClient` 来使用开源模型！

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

就这么简单！只需将 `from openai import OpenAI` 替换为 `from huggingface_hub import InferenceClient`，再将 `client = OpenAI(...)` 替换为 `client = InferenceClient(...)`。你可以通过将模型 ID 作为 `model` 参数传入，选择 Hugging Face Hub 上的任意 LLM 模型。这里有一个[支持模型列表](https://huggingface.co/models?pipeline_tag=text-generation&other=conversational,text-generation-inference&sort=trending)。身份验证时，可以将有效的[用户访问令牌](https://huggingface.co/settings/tokens)作为 `api_key` 传入，或使用 `huggingface_hub` 完成身份验证（参阅[身份验证指南](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)）。

所有输入参数和输出格式都严格一致。特别是，你可以传入 `stream=True`，在生成令牌时逐个接收。你还可以使用 [`AsyncInferenceClient`]，通过 `asyncio` 运行推理：

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

你可能会问，为什么要使用 [`InferenceClient`]，而不是 OpenAI 的客户端？原因有以下几点：
1. [`InferenceClient`] 针对 Hugging Face 服务进行了配置。使用 Inference Providers 运行模型时，无需提供 `base_url`。如果本机已经正确登录，也无需提供 `token` 或 `api_key`。
2. [`InferenceClient`] 同时针对 Text-Generation-Inference (TGI) 和 `transformers` 框架进行了适配，因此可以持续跟进最新更新。
3. [`InferenceClient`] 集成了 Inference Endpoints 服务，可以更轻松地启动 Inference Endpoint、检查其状态并运行推理。更多详情请参阅 [Inference Endpoints](./inference_endpoints.md) 指南。

> [!TIP]
> `InferenceClient.chat.completions.create` 只是 `InferenceClient.chat_completion` 的别名。更多详情请参阅 [`~InferenceClient.chat_completion`] 的包参考文档。初始化客户端时的 `base_url` 和 `api_key` 参数也是 `model` 和 `token` 的别名。提供这些别名是为了降低从 `OpenAI` 切换到 `InferenceClient` 的摩擦。

## 函数调用

函数调用允许 LLM 与外部工具（例如定义好的函数或 API）交互，从而可以轻松构建适用于具体场景和现实任务的应用。
`InferenceClient` 实现了与 OpenAI Chat Completions API 相同的工具调用接口。下面是使用 [Novita](https://novita.ai/) 作为推理服务商进行工具调用的简单示例：

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

client = InferenceClient(provider="novita")

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
> 请参阅服务商的文档，确认哪些模型支持函数/工具调用。

## 结构化输出和 JSON 模式

InferenceClient 支持 JSON 模式，可生成语法有效的 JSON 响应；也支持结构化输出，可生成符合指定 schema 的响应。JSON 模式提供机器可读的数据，但不保证严格结构；结构化输出则同时保证 JSON 有效，并遵循预定义 schema，适合可靠的下游处理。

我们遵循 OpenAI API 关于 JSON 模式和结构化输出的规范。你可以通过 `response_format` 参数启用它们。下面是使用 [Cerebras](https://www.cerebras.ai/) 作为推理服务商进行结构化输出的示例：

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
> 请参阅服务商的文档，确认哪些模型支持结构化输出和 JSON 模式。

## 异步客户端

我们还提供了基于 `asyncio` 和 `httpx2` 的异步客户端。所有异步 API 端点都可以通过 [`AsyncInferenceClient`] 使用。它的初始化方式和 API 与仅支持同步调用的版本完全相同。

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

有关 `asyncio` 模块的更多信息，请参阅[官方文档](https://docs.python.org/3/library/asyncio.html)。

## MCP 客户端

`huggingface_hub` 库现在包含实验性的 [`MCPClient`]，旨在让大型语言模型（LLM）能够通过[模型上下文协议](https://modelcontextprotocol.io)（MCP）与外部工具交互。该客户端扩展了 [`AsyncInferenceClient`]，可以无缝集成工具使用能力。

[`MCPClient`] 连接到暴露工具的 MCP 服务器（可以是本地 `stdio` 脚本，也可以是远程 `http`/`sse` 服务）。它通过 [`AsyncInferenceClient`] 将这些工具提供给 LLM。如果 LLM 决定使用某个工具，[`MCPClient`] 会管理向 MCP 服务器发送的执行请求，并将工具输出传回 LLM，通常还会实时流式传输结果。

下面的示例使用 [Novita](https://novita.ai/) 推理服务商提供的 [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) 模型。随后，我们添加一个远程 MCP 服务器；这里使用的是一个 SSE 服务器，它向 LLM 提供 Flux 图像生成工具。

```python
import os

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient


async def main():
    async with MCPClient(
        provider="novita",
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


为了进一步简化开发，我们还提供了更高层的 [`Agent`] 类。这个“Tiny Agent”通过管理聊天循环和状态，简化了对话式 Agent 的创建，本质上是 [`MCPClient`] 外面的一层封装。它直接建立在 [`MCPClient`] 之上的简单 while 循环。你可以直接从命令行运行这些 Agent：


```bash
# install latest version of huggingface_hub with the mcp extra
pip install -U huggingface_hub[mcp]
# Run an agent that uses the Flux image generation tool
tiny-agents run julien-c/flux-schnell-generator

```

启动后，Agent 会加载并列出从已连接 MCP 服务器发现的工具，然后就可以接收你的提示了！

## 高级技巧

在上面的部分中，我们介绍了 [`InferenceClient`] 的主要功能。下面进一步了解一些高级技巧。

### 计费

作为 HF 用户，你每月会获得积分，可用于在 Hub 上通过各种服务商运行推理。积分额度取决于账户类型（Free、PRO 或 Enterprise Hub）。每次推理请求都会按照服务商的价格表计费。默认情况下，请求会计费到你的个人账户。不过，只需将 `bill_to="<your_org_name>"` 传给 `InferenceClient`，就可以将请求费用计入你所属的组织。要使用此功能，你的组织必须订阅 Enterprise Hub。有关计费的更多详情，请参阅[此指南](https://huggingface.co/docs/api-inference/pricing#features-using-inference-providers)。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="fal-ai", bill_to="openai")
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```

注意，不能向其他用户或不属于你的组织收取费用。如果希望赠送积分给其他人，必须与对方创建一个共同组织。


### 超时

推理调用可能需要较长时间。默认情况下，[`InferenceClient`] 会“无限期”等待推理完成。如果希望更好地控制工作流，可以将 `timeout` 参数设置为具体的秒数。超时后会抛出 [`InferenceTimeoutError`]，你可以捕获该异常：

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### 二进制输入

有些任务需要二进制输入，例如图像或音频文件。在这种情况下，[`InferenceClient`] 会尽可能灵活地接受不同类型的输入：
- 原始 `bytes`
- 以二进制方式打开的类文件对象（`with open("audio.flac", "rb") as f: ...`）
- 指向本地文件的路径（`str` 或 `Path`）
- 指向远程文件的 URL（`str`，例如 `https://...`）。这种情况下，文件会先下载到本地，然后再发送到 API。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```
