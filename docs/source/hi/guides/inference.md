<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Server पर Inference चलाएँ

Inference वह प्रक्रिया है जिसमें एक trained model का उपयोग करके नए डेटा पर predictions प्राप्त किए जाते हैं। इस प्रक्रिया में काफ़ी computing संसाधन लग सकते हैं, इसलिए इसे किसी dedicated या बाहरी service पर चलाना एक उपयोगी विकल्प हो सकता है।
`huggingface_hub` library, Hugging Face Hub पर होस्ट किए गए models के लिए कई services पर inference चलाने का एक साझा interface प्रदान करती है:

1.  [Inference Providers](https://huggingface.co/docs/inference-providers/index): हमारे serverless inference partners की मदद से सैकड़ों machine learning models तक आसान और एकीकृत पहुँच। यह नया तरीका हमारे पिछले Serverless Inference API पर आधारित है और विश्वस्तरीय providers की मदद से अधिक models, बेहतर performance और अधिक विश्वसनीयता प्रदान करता है। समर्थित providers की सूची के लिए [दस्तावेज़](https://huggingface.co/docs/inference-providers/index#partners) देखें।
2.  [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index): models को आसानी से production में deploy करने के लिए एक उत्पाद। Hugging Face आपकी पसंद के cloud provider पर dedicated और पूरी तरह से managed infrastructure में inference चलाता है।
3.  Local endpoints: आप client को local endpoints से जोड़कर [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm), [LiteLLM](https://docs.litellm.ai/docs/simple_proxy) या [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) जैसे local inference servers के साथ भी inference चला सकते हैं।

> [!TIP]
> [`InferenceClient`] एक Python client है जो हमारे APIs को HTTP calls भेजता है। यदि आप अपने पसंदीदा tool
> (curl, postman,...) से सीधे HTTP calls करना चाहते हैं, तो [Inference Providers](https://huggingface.co/docs/inference-providers/index)
> या [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) के दस्तावेज़ देखें।
>
> web development के लिए एक [JS client](https://huggingface.co/docs/huggingface.js/inference/README) उपलब्ध है।
> यदि आपकी रुचि game development में है, तो हमारा [C# project](https://github.com/huggingface/unity-api) देख सकते हैं।

## शुरुआत करें

चलिए एक text-to-image task से शुरुआत करते हैं:

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

ऊपर दिए गए उदाहरण में हमने एक third-party provider, [Replicate](https://replicate.com/), के साथ [`InferenceClient`] को initialize किया। किसी provider का उपयोग करते समय आपको वह model निर्दिष्ट करना होगा जिसे आप इस्तेमाल करना चाहते हैं। model id, Hugging Face Hub पर मौजूद model की id होनी चाहिए, न कि third-party provider पर मौजूद model की id।
इस उदाहरण में हमने एक text prompt से image बनाई। लौटाई गई value एक `PIL.Image` object है, जिसे फ़ाइल में सहेजा जा सकता है। अधिक जानकारी के लिए [`~InferenceClient.text_to_image`] के दस्तावेज़ देखें।

अब [`~InferenceClient.chat_completion`] API का एक उदाहरण देखते हैं। यह task, messages की सूची से जवाब तैयार करने के लिए LLM का उपयोग करता है:

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

ऊपर दिए गए उदाहरण में हमने एक third-party provider ([Together AI](https://www.together.ai/)) का उपयोग किया और वह model निर्दिष्ट किया जिसे हम इस्तेमाल करना चाहते हैं (`"meta-llama/Meta-Llama-3-8B-Instruct"`)। फिर हमने जवाब तैयार करने के लिए messages की एक सूची दी (यहाँ केवल एक सवाल) और API को एक अतिरिक्त parameter (`max_token=100`) दिया। output एक `ChatCompletionOutput` object है जो OpenAI specification का पालन करता है। तैयार किया गया content, `output.choices[0].message.content` से प्राप्त किया जा सकता है। अधिक जानकारी के लिए [`~InferenceClient.chat_completion`] के दस्तावेज़ देखें।


> [!WARNING]
> API को सरल रखने के लिए बनाया गया है। सभी parameters और options अंतिम उपयोगकर्ता के लिए उपलब्ध या वर्णित नहीं हैं।
> प्रत्येक task के लिए उपलब्ध सभी parameters के बारे में अधिक जानने के लिए
> [यह पेज](https://huggingface.co/docs/api-inference/detailed_parameters) देखें।

### किसी विशेष provider का उपयोग करें

यदि आप किसी विशेष provider का उपयोग करना चाहते हैं, तो client को initialize करते समय उसे निर्दिष्ट कर सकते हैं। default value "auto" है, जो model के लिए उपलब्ध providers में से पहला provider चुनती है। इन providers का क्रम https://hf.co/settings/inference-providers पर उपयोगकर्ता द्वारा तय किए गए क्रम के अनुसार होता है। समर्थित providers की सूची के लिए [समर्थित providers और tasks](#supported-providers-and-tasks) सेक्शन देखें।

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="replicate", api_key="my_replicate_api_key")
```

### किसी विशेष model का उपयोग करें

यदि आप किसी विशेष model का उपयोग करना चाहते हैं, तो उसे parameter के रूप में या सीधे instance के स्तर पर निर्दिष्ट कर सकते हैं:

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
> "hf-inference" provider का उपयोग करते समय, प्रत्येक task के लिए Hub पर उपलब्ध 10 लाख से अधिक models में से एक model सुझाया जाता है।
> हालाँकि, यह सुझाव समय के साथ बदल सकता है, इसलिए model चुन लेने के बाद उसे स्पष्ट रूप से सेट करना बेहतर है।
> third-party providers के लिए आपको हमेशा ऐसा model निर्दिष्ट करना होगा जो उस provider के साथ compatible हो।
>
> Inference Providers के ज़रिए उपलब्ध models को देखने के लिए Hub के [Models](https://huggingface.co/models?inference=warm) पेज पर जाएँ।

### Inference Endpoints का उपयोग करें

ऊपर दिए गए उदाहरण inference providers का उपयोग करते हैं। ये जल्दी prototype बनाने और चीज़ों को आज़माने के लिए बहुत उपयोगी हैं।
जब आप अपने model को production में deploy करने के लिए तैयार हों, तो आपको dedicated infrastructure की आवश्यकता होगी।
यहीं [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) काम आता है। इसकी मदद से आप किसी भी model को deploy करके
उसे private API के रूप में उपलब्ध करा सकते हैं। deploy करने के बाद आपको एक URL मिलेगा, जिससे जुड़ने के लिए आप पहले वाला ही
code इस्तेमाल कर सकते हैं; केवल `model` parameter बदलना होगा:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# or
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

ध्यान दें कि आप URL और provider दोनों एक साथ निर्दिष्ट नहीं कर सकते। URLs का उपयोग सीधे deployed endpoints से जुड़ने के लिए होता है।

### local endpoints का उपयोग करें

आप अपनी मशीन पर चल रहे local inference servers (llama.cpp, vllm, litellm server, TGI, mlx आदि) के साथ chat completion चलाने के लिए [`InferenceClient`] का उपयोग कर सकते हैं। API को OpenAI API के साथ compatible होना चाहिए।

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
> OpenAI Python client की तरह, [`InferenceClient`] का उपयोग किसी भी OpenAI REST API-compatible endpoint के साथ Chat Completion inference चलाने के लिए किया जा सकता है।

### Authentication करें

Authentication दो तरीकों से किया जा सकता है:

**Hugging Face के ज़रिए routing** : third-party providers तक पहुँचने के लिए Hugging Face को proxy के रूप में इस्तेमाल करें। calls हमारी provider keys का उपयोग करके Hugging Face के infrastructure से होकर जाएँगी और उपयोग का बिल सीधे आपके Hugging Face खाते में जुड़ेगा।

आप [User Access Token](https://huggingface.co/docs/hub/security-tokens) से authenticate कर सकते हैं। `api_key` parameter का उपयोग करके अपना Hugging Face token सीधे दे सकते हैं:

```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="hf_****"  # Your HF token
)
```

यदि आप `api_key` *नहीं* देते हैं, तो client आपकी मशीन पर स्थानीय रूप से सहेजा गया token ढूँढ़कर उसका उपयोग करने की कोशिश करेगा। ऐसा आम तौर पर तब होता है जब आपने पहले login किया हो। login की जानकारी के लिए [Authentication मार्गदर्शिका](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) देखें।

```python
>>> client = InferenceClient(
    provider="replicate",
    token="hf_****"  # Your HF token
)
```

**provider तक सीधी पहुँच**: provider की service से सीधे जुड़ने के लिए अपनी API key का उपयोग करें:
```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="r8_****"  # Your Replicate API key
)
```

अधिक जानकारी के लिए [Inference Providers के pricing दस्तावेज़](https://huggingface.co/docs/inference-providers/pricing#routed-requests-vs-direct-calls) देखें।

## समर्थित providers और tasks [[supported-providers-and-tasks]]

[`InferenceClient`] का उद्देश्य किसी भी provider पर Hugging Face models के साथ inference चलाने के लिए सबसे आसान interface प्रदान करना है। इसका सरल API सबसे आम tasks को support करता है। नीचे दी गई तालिका बताती है कि कौन-से providers किन tasks को support करते हैं:

| कार्य                                                | Baseten | Cerebras | Cohere | DeepInfra | fal-ai | Featherless AI | Fireworks AI | Groq | HF Inference | Novita AI | Nscale | OVHcloud AI Endpoints | Public AI | Replicate | Scaleway | Together | Wavespeed | Zai |
| --------------------------------------------------- | ------- | -------- | ------ | --------- | ------ | -------------- | ------------ | ---- | ------------ | --------- | ------ | --------------------- | --------- | --------- | -------- | -------- | --------- | --- |
| [`~InferenceClient.audio_classification`]           | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.audio_to_audio`]                 | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.automatic_speech_recognition`]   | ❌      | ❌        | ❌      | ✅         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ✅         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.chat_completion`]                | ✅      | ✅        | ✅      | ✅         | ❌      | ✅              | ✅            | ✅    | ✅            | ✅         | ✅      | ✅                     | ✅         | ❌         | ✅        | ✅        | ❌         | ✅   |
| [`~InferenceClient.document_question_answering`]    | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.feature_extraction`]             | ❌      | ❌        | ❌      | ✅         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ✅        | ✅        | ❌         | ❌   |
| [`~InferenceClient.fill_mask`]                      | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.image_classification`]           | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.image_segmentation`]             | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.image_to_image`]                 | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ✅         | ❌        | ✅        | ✅         | ❌   |
| [`~InferenceClient.image_to_video`]                 | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ❌            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ✅        | ✅         | ❌   |
| [`~InferenceClient.image_to_text`]                  | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.object_detection`]               | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.question_answering`]             | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.sentence_similarity`]            | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.summarization`]                  | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.table_question_answering`]       | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.text_classification`]            | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.text_generation`]                | ❌      | ❌        | ❌      | ✅         | ❌      | ✅              | ❌            | ❌    | ✅            | ✅         | ❌      | ❌                     | ❌         | ❌         | ❌        | ✅        | ❌         | ❌   |
| [`~InferenceClient.text_to_image`]                  | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ✅      | ❌                     | ❌         | ✅         | ❌        | ✅        | ✅         | ✅   |
| [`~InferenceClient.text_to_speech`]                 | ❌      | ❌        | ❌      | ✅         | ✅      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ✅         | ❌        | ✅        | ❌         | ❌   |
| [`~InferenceClient.text_to_video`]                  | ❌      | ❌        | ❌      | ❌         | ✅      | ❌              | ❌            | ❌    | ❌            | ✅         | ❌      | ❌                     | ❌         | ✅         | ❌        | ✅        | ✅         | ❌   |
| [`~InferenceClient.tabular_classification`]         | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.tabular_regression`]             | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.token_classification`]           | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.translation`]                    | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.visual_question_answering`]      | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.zero_shot_image_classification`] | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |
| [`~InferenceClient.zero_shot_classification`]       | ❌      | ❌        | ❌      | ❌         | ❌      | ❌              | ❌            | ❌    | ✅            | ❌         | ❌      | ❌                     | ❌         | ❌         | ❌        | ❌        | ❌         | ❌   |

> [!TIP]
> प्रत्येक task के बारे में अधिक जानने के लिए [Tasks](https://huggingface.co/tasks) पेज देखें।

## OpenAI के साथ compatibility

`chat_completion` task, [OpenAI के Python client](https://github.com/openai/openai-python) की syntax का पालन करता है। आपके लिए इसका क्या मतलब है? यदि आप `OpenAI` के APIs का उपयोग करते आए हैं, तो code की केवल 2 पंक्तियाँ बदलकर open-source models के साथ काम करने के लिए `huggingface_hub.InferenceClient` अपना सकते हैं!

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

बस इतना ही! आपको केवल `from openai import OpenAI` की जगह `from huggingface_hub import InferenceClient` और `client = OpenAI(...)` की जगह `client = InferenceClient(...)` लिखना है। आप Hugging Face Hub से कोई भी LLM model चुन सकते हैं और उसकी model id को `model` parameter में दे सकते हैं। समर्थित models की [सूची यहाँ है](https://huggingface.co/models?pipeline_tag=text-generation&other=conversational,text-generation-inference&sort=trending)। authentication के लिए एक मान्य [User Access Token](https://huggingface.co/settings/tokens) को `api_key` के रूप में दें या `huggingface_hub` से authenticate करें ([authentication मार्गदर्शिका](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) देखें)।

सभी input parameters और output format बिल्कुल समान हैं। खास तौर पर, tokens बनते ही उन्हें प्राप्त करने के लिए आप `stream=True` दे सकते हैं। `asyncio` के साथ inference चलाने के लिए आप [`AsyncInferenceClient`] का भी उपयोग कर सकते हैं:

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

आप सोच सकते हैं कि OpenAI के client की जगह [`InferenceClient`] का उपयोग क्यों करें? इसके कुछ कारण हैं:
1. [`InferenceClient`], Hugging Face services के लिए configured है। Inference Providers के साथ models चलाने के लिए आपको `base_url` देने की ज़रूरत नहीं है। यदि आपकी मशीन पर पहले से सही तरीके से login किया हुआ है, तो `token` या `api_key` भी देने की ज़रूरत नहीं है।
2. [`InferenceClient`] को Text-Generation-Inference (TGI) और `transformers`, दोनों frameworks के अनुरूप बनाया गया है, इसलिए यह उनके नवीनतम updates के साथ तालमेल बनाए रखता है।
3. [`InferenceClient`] हमारी Inference Endpoints service के साथ integrated है, जिससे Inference Endpoint शुरू करना, उसकी स्थिति जाँचना और उस पर inference चलाना आसान होता है। अधिक जानकारी के लिए [Inference Endpoints](./inference_endpoints.md) मार्गदर्शिका देखें।

> [!TIP]
> `InferenceClient.chat.completions.create`, `InferenceClient.chat_completion` का ही एक alias है। अधिक जानकारी के लिए [`~InferenceClient.chat_completion`] का package reference देखें। client का instance बनाते समय `base_url` और `api_key` parameters भी क्रमशः `model` और `token` के aliases हैं। ये aliases, `OpenAI` से `InferenceClient` पर जाना आसान बनाने के लिए दिए गए हैं।

## Function Calling का उपयोग करें

Function calling की मदद से LLMs बाहरी tools, जैसे परिभाषित functions या APIs, के साथ काम कर सकते हैं। इससे उपयोगकर्ता अपने खास उपयोग और वास्तविक दुनिया के कार्यों के लिए आसानी से applications बना सकते हैं।
`InferenceClient`, OpenAI Chat Completions API वाला ही tool calling interface लागू करता है। यहाँ [Novita](https://novita.ai/) को inference provider के रूप में इस्तेमाल करके tool calling का एक सरल उदाहरण दिया गया है:

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
> Function/Tool Calling के लिए providers किन models को support करते हैं, यह जाँचने के लिए उनके दस्तावेज़ देखें।

## Structured Outputs और JSON Mode

InferenceClient, सही JSON syntax वाले responses के लिए JSON mode और किसी schema का पालन करने वाले responses के लिए Structured Outputs को support करता है। JSON mode बिना किसी सख्त संरचना के machine-readable डेटा देता है, जबकि Structured Outputs यह सुनिश्चित करते हैं कि JSON मान्य हो और पहले से तय schema का पालन करे, ताकि आगे की processing विश्वसनीय हो।

हम JSON mode और Structured Outputs, दोनों के लिए OpenAI API specifications का पालन करते हैं। आप इन्हें `response_format` argument से सक्षम कर सकते हैं। यहाँ [Cerebras](https://www.cerebras.ai/) को inference provider के रूप में इस्तेमाल करके Structured Outputs का एक उदाहरण दिया गया है:

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
> Structured Outputs और JSON Mode के लिए providers किन models को support करते हैं, यह जाँचने के लिए उनके दस्तावेज़ देखें।

## Async client का उपयोग करें

client का एक async संस्करण भी उपलब्ध है, जो `asyncio` और `httpx` पर आधारित है। सभी async API endpoints, [`AsyncInferenceClient`] के ज़रिए उपलब्ध हैं। इसका initialization और APIs, sync-only संस्करण के बिल्कुल समान हैं।

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

`asyncio` module के बारे में अधिक जानकारी के लिए [आधिकारिक दस्तावेज़](https://docs.python.org/3/library/asyncio.html) देखें।

## MCP Client का उपयोग करें

`huggingface_hub` library में अब एक प्रायोगिक [`MCPClient`] शामिल है, जो Large Language Models (LLMs) को [Model Context Protocol](https://modelcontextprotocol.io) (MCP) के ज़रिए बाहरी tools के साथ काम करने की क्षमता देता है। यह client, [`AsyncInferenceClient`] का विस्तार करके tools का उपयोग आसानी से जोड़ता है।

[`MCPClient`] उन MCP servers से जुड़ता है जो tools उपलब्ध कराते हैं। ये local `stdio` scripts या remote `http`/`sse` services हो सकती हैं। यह इन tools को LLM तक ([`AsyncInferenceClient`] के ज़रिए) पहुँचाता है। यदि LLM किसी tool का उपयोग करने का निर्णय लेता है, तो [`MCPClient`], MCP server को भेजे जाने वाले execution request को संभालता है और tool का output वापस LLM तक पहुँचाता है। इसमें अक्सर results की real-time streaming भी होती है।

निम्न उदाहरण में हम [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model का उपयोग [Novita](https://novita.ai/) inference provider के ज़रिए करते हैं। फिर हम एक remote MCP server जोड़ते हैं। यहाँ यह एक SSE server है जो LLM को Flux image generation tool उपलब्ध कराता है।

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


development को और आसान बनाने के लिए हम एक higher-level [`Agent`] class भी प्रदान करते हैं। यह 'Tiny Agent', chat loop और state को संभालकर बातचीत करने वाले Agents बनाना आसान करता है और मूल रूप से [`MCPClient`] के wrapper की तरह काम करता है। इसे [`MCPClient`] के ऊपर बने एक सरल while loop के रूप में डिज़ाइन किया गया है। आप इन Agents को सीधे command line से चला सकते हैं:


```bash
# install latest version of huggingface_hub with the mcp extra
pip install -U huggingface_hub[mcp]
# Run an agent that uses the Flux image generation tool
tiny-agents run julien-c/flux-schnell-generator

```

शुरू होने पर Agent लोड होगा और जुड़े हुए MCP servers पर मिले tools की सूची दिखाएगा। इसके बाद वह आपके prompts के लिए तैयार है!

## उन्नत सुझाव

ऊपर के सेक्शन में हमने [`InferenceClient`] की मुख्य बातें देखीं। अब कुछ उन्नत सुझाव देखते हैं।

### Billing का प्रबंधन करें

HF उपयोगकर्ता के रूप में आपको Hub पर अलग-अलग providers के ज़रिए inference चलाने के लिए हर महीने credits मिलते हैं। credits की मात्रा आपके खाते के प्रकार (Free, PRO या Enterprise Hub) पर निर्भर करती है। प्रत्येक inference request के लिए provider की pricing table के अनुसार शुल्क लिया जाता है। default रूप से requests का बिल आपके व्यक्तिगत खाते में जुड़ता है। हालाँकि, आप `InferenceClient` को केवल `bill_to="<your_org_name>"` देकर requests का बिल उस organization में जोड़ सकते हैं जिसके आप सदस्य हैं। इसके लिए आपके organization के पास Enterprise Hub का subscription होना चाहिए। billing के बारे में अधिक जानकारी के लिए [यह मार्गदर्शिका](https://huggingface.co/docs/api-inference/pricing#features-using-inference-providers) देखें।

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="fal-ai", bill_to="openai")
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```

ध्यान दें कि किसी दूसरे उपयोगकर्ता या ऐसे organization से शुल्क लेना संभव नहीं है जिसके आप सदस्य नहीं हैं। यदि आप किसी और को कुछ credits देना चाहते हैं, तो आपको उनके साथ एक साझा organization बनाना होगा।


### Timeout सेट करें

Inference calls में काफ़ी समय लग सकता है। default रूप से [`InferenceClient`], inference पूरा होने तक "अनिश्चित समय" तक प्रतीक्षा करेगा। यदि आप अपने workflow पर अधिक नियंत्रण चाहते हैं, तो `timeout` parameter में सेकंड में एक निश्चित अवधि सेट कर सकते हैं। यदि यह अवधि बीत जाती है, तो [`InferenceTimeoutError`] उठाया जाता है, जिसे आप अपने code में catch कर सकते हैं:

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### Binary inputs दें

कुछ tasks में binary inputs की ज़रूरत होती है, जैसे images या audio फ़ाइलों के साथ काम करते समय। ऐसे मामलों में [`InferenceClient`]
अधिक से अधिक input प्रकारों को स्वीकार करने की कोशिश करता है:
- raw `bytes`
- binary mode में खोला गया file-like object (`with open("audio.flac", "rb") as f: ...`)
- किसी local फ़ाइल का path (`str` या `Path`)
- किसी remote फ़ाइल का URL (`str`) (जैसे `https://...`)। इस स्थिति में API को भेजने से पहले फ़ाइल को स्थानीय रूप से
डाउनलोड किया जाएगा।

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```
