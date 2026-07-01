<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hub पर खोजें

इस ट्यूटोरियल में, आप सीखेंगे कि `huggingface_hub` का उपयोग करके Hub पर Models, Datasets और Spaces कैसे खोजें।

## Repositories को कैसे सूचीबद्ध करें ?

`huggingface_hub` library में Hub के साथ इंटरैक्ट करने के लिए एक HTTP client [`HfApi`] शामिल है।
अन्य चीज़ों के अलावा, यह Hub पर संग्रहीत Models, Datasets और Spaces को सूचीबद्ध कर सकता है:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[`list_models`] का आउटपुट Hub पर संग्रहीत Models का एक iterator होता है।

इसी तरह, आप Datasets को सूचीबद्ध करने के लिए [`list_datasets`] और Spaces को सूचीबद्ध करने के लिए [`list_spaces`] का उपयोग कर सकते हैं।

## Repositories को कैसे फ़िल्टर करें ?

Repositories को सूचीबद्ध करना तो बढ़िया है, लेकिन अब शायद आप अपनी खोज को फ़िल्टर करना चाहें।
list helpers में कई attributes होते हैं, जैसे:
- `filter`
- `author`
- `search`
- `num_parameters`
- ...

चलिए एक उदाहरण देखते हैं जिसमें Hub पर मौजूद वे सभी Models प्राप्त किए जाते हैं जो image classification करते हैं, imagenet dataset पर ट्रेन किए गए हैं और PyTorch के साथ चलते हैं।

```py
models = hf_api.list_models(filter=["image-classification", "pytorch", "imagenet"])
```

आप Hub UI जैसी ही range syntax का उपयोग करके Models को parameter count के आधार पर भी फ़िल्टर कर सकते हैं:

```py
models = hf_api.list_models(num_parameters="min:6B,max:128B")
```

फ़िल्टर करते समय, आप Models को sort भी कर सकते हैं और केवल top results ही ले सकते हैं। उदाहरण के लिए,
नीचे दिया गया उदाहरण Hub पर सबसे ज़्यादा डाउनलोड किए गए top 5 Datasets प्राप्त करता है:

```py
>>> list(list_datasets(sort="downloads", limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



Hub पर उपलब्ध filters को एक्सप्लोर करने के लिए, अपने browser में [models](https://huggingface.co/models) और [datasets](https://huggingface.co/datasets) पेज खोलें,
कुछ parameters खोजें और URL में मौजूद values देखें।

## CLI का उपयोग करना

आप `hf` command-line interface का उपयोग करके भी Models, Datasets और Spaces को सूचीबद्ध और खोज सकते हैं:

```bash
# Models सूचीबद्ध करें
>>> hf models ls --search "llama" --sort downloads --limit 5

# Datasets सूचीबद्ध करें
>>> hf datasets ls --author Qwen

# Spaces सूचीबद्ध करें
>>> hf spaces ls --search "3d"

# किसी विशिष्ट model की जानकारी प्राप्त करें
>>> hf models info Lightricks/LTX-2
```

अधिक जानकारी के लिए, [CLI गाइड](./cli.md#hf-models) देखें।
