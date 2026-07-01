<!--⚠️ ध्यान दें कि यह फ़ाइल Markdown में है लेकिन इसमें हमारे doc-builder के लिए विशेष syntax (MDX जैसी) है जो आपके Markdown viewer में सही तरह render नहीं हो सकती।
-->

# Hub से फ़ाइलें डाउनलोड करें

`huggingface_hub` लाइब्रेरी Hub पर stored repositories से फ़ाइलें डाउनलोड करने के लिए functions प्रदान करती है। आप इन functions को अलग से भी इस्तेमाल कर सकते हैं, या इन्हें अपनी लाइब्रेरी में integrate करके अपने users के लिए Hub के साथ काम करना आसान बना सकते हैं। यह गाइड आपको निम्नलिखित करना सिखाएगी:

* एकल फ़ाइल डाउनलोड और cache करना।
* पूरी repository डाउनलोड और cache करना।
* फ़ाइलें किसी local folder में डाउनलोड करना।

## एकल फ़ाइल डाउनलोड करें

[`hf_hub_download`] function Hub से फ़ाइलें डाउनलोड करने के लिए मुख्य function है। यह remote file को डाउनलोड करता है, उसे disk पर cache करता है (version की जानकारी के साथ), और उसका local file path return करता है।

> [!TIP]
> जो filepath return होता है वह HF के local cache की ओर इशारा करता है। इसलिए इस फ़ाइल को modify न करें, वरना cache corrupt हो सकता है। फ़ाइलें किस तरह cache होती हैं, यह जानने के लिए हमारी [caching guide](./manage-cache) देखें।

### Latest version से

`repo_id`, `repo_type` और `filename` parameters की मदद से डाउनलोड होने वाली फ़ाइल चुनें। डिफ़ॉल्ट रूप से फ़ाइल को `model` repo का हिस्सा माना जाता है।

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# किसी dataset से डाउनलोड करें
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### किसी specific version से

डिफ़ॉल्ट रूप से `main` branch का latest version डाउनलोड होता है। लेकिन कई बार आप किसी specific version की फ़ाइल चाहते हैं — जैसे किसी खास branch, PR, tag या commit hash से। इसके लिए `revision` parameter का उपयोग करें:

```python
# `v1.0` tag से डाउनलोड करें
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# `test-branch` branch से डाउनलोड करें
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Pull Request #3 से डाउनलोड करें
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# किसी specific commit hash से डाउनलोड करें
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**नोट:** Commit hash का उपयोग करते समय 7-character के short hash की बजाय पूरा full-length hash देना ज़रूरी है।

### Download URL बनाएं

अगर आप किसी repo से फ़ाइल डाउनलोड करने के लिए URL खुद बनाना चाहते हैं, तो [`hf_hub_url`] का उपयोग करें जो एक URL return करता है। यह [`hf_hub_download`] के अंदर internally भी इस्तेमाल होता है।

## पूरी repository डाउनलोड करें

[`snapshot_download`] किसी दिए गए revision पर पूरी repository डाउनलोड करता है। यह internally [`hf_hub_download`] का उपयोग करता है, इसलिए सभी डाउनलोड की गई फ़ाइलें आपकी local disk पर cache भी होती हैं। प्रक्रिया तेज़ करने के लिए downloads concurrently किए जाते हैं।

पूरी repository डाउनलोड करने के लिए बस `repo_id` और `repo_type` pass करें:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# या किसी dataset से
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[`snapshot_download`] डिफ़ॉल्ट रूप से latest revision डाउनलोड करता है। कोई specific revision चाहिए तो `revision` parameter का उपयोग करें:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### डाउनलोड के लिए फ़ाइलें filter करें

[`snapshot_download`] पूरी repository आसानी से डाउनलोड कर देता है, लेकिन हमेशा पूरा content डाउनलोड करना ज़रूरी नहीं होता। मसलन, अगर आप सिर्फ `.safetensors` weights इस्तेमाल करने वाले हैं, तो सभी `.bin` फ़ाइलें डाउनलोड करने से बचना चाहेंगे। इसके लिए `allow_patterns` और `ignore_patterns` parameters काम आते हैं।

ये parameters एकल pattern या patterns की list स्वीकार करते हैं। Patterns, Standard Wildcards (globbing patterns) हैं जैसा [यहाँ](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) बताया गया है। Pattern matching [`fnmatch`](https://docs.python.org/3/library/fnmatch.html) पर आधारित है।

उदाहरण के लिए, केवल JSON configuration files डाउनलोड करने के लिए `allow_patterns` इस्तेमाल करें:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

इसके विपरीत, `ignore_patterns` कुछ फ़ाइलों को डाउनलोड से बाहर रख सकता है। नीचे का उदाहरण `.msgpack` और `.h5` extensions को ignore करता है:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

आप दोनों को मिलाकर डाउनलोड को बिल्कुल सटीक तरीके से filter भी कर सकते हैं। यहाँ `vocab.json` को छोड़कर सभी json और markdown फ़ाइलें डाउनलोड करने का उदाहरण है:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## फ़ाइलें किसी local folder में डाउनलोड करें

डिफ़ॉल्ट रूप से हम Hub से फ़ाइलें डाउनलोड करने के लिए [cache system](./manage-cache) इस्तेमाल करने की सलाह देते हैं। [`hf_hub_download`] और [`snapshot_download`] में `cache_dir` parameter देकर, या [`HF_HOME`](../package_reference/environment_variables#hf_home) environment variable सेट करके custom cache location तय की जा सकती है।

लेकिन अगर आपको फ़ाइलें किसी खास folder में डाउनलोड करनी हों, तो download function में `local_dir` parameter pass करें। यह `git` command जैसा workflow देता है। डाउनलोड की गई फ़ाइलें specified folder के अंदर अपनी मूल file structure बनाए रखती हैं। उदाहरण के लिए, अगर `filename="data/train.csv"` और `local_dir="path/to/folder"` है, तो resulting filepath `"path/to/folder/data/train.csv"` होगा।

आपकी local directory के root पर एक `.cache/huggingface/` folder बनता है जिसमें डाउनलोड की गई फ़ाइलों की metadata होती है। इससे पहले से up-to-date फ़ाइलें दोबारा डाउनलोड नहीं होतीं। अगर metadata बदल गई हो तो नया file version डाउनलोड होता है। इस तरह `local_dir` केवल latest changes pull करने के लिए optimized रहता है।

डाउनलोड पूरा होने के बाद, अगर ज़रूरत न हो तो `.cache/huggingface/` folder safely हटाया जा सकता है। हालाँकि, ध्यान रखें कि इस folder के बिना script दोबारा चलाने पर metadata न होने की वजह से recovery में अधिक समय लग सकता है। आपका local data पूरी तरह सुरक्षित रहेगा।

> [!TIP]
> Hub पर changes commit करते समय `.cache/huggingface/` folder की चिंता न करें! यह folder `git` और [`upload_folder`] दोनों द्वारा automatically ignore किया जाता है।

## CLI से डाउनलोड करें

Terminal से सीधे Hub से फ़ाइलें डाउनलोड करने के लिए `hf download` command का उपयोग करें। यह internally ऊपर बताए गए [`hf_hub_download`] और [`snapshot_download`] helpers का ही उपयोग करता है और terminal पर returned path print करता है।

```bash
>>> hf download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

एक साथ कई फ़ाइलें डाउनलोड की जा सकती हैं — इसमें progress bar दिखती है और snapshot path return होता है जहाँ फ़ाइलें मौजूद हैं:

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

एकल `hf://` URI से भी repo (और वैकल्पिक रूप से revision और file) को point कर सकते हैं। URI का grammar है `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]` (पूरे syntax के लिए [HF URIs reference](../package_reference/hf_uris) देखें) — यह `--repo-type` और `--revision` options की जगह लेता है, जिन्हें इसके साथ set नहीं किया जा सकता:

```bash
# किसी dataset की फ़ाइल को दिए गए revision पर डाउनलोड करें
>>> hf download hf://datasets/google/fleurs@refs/pr/1/fleurs.py

# कोई subfolder डाउनलोड करें (trailing slash ज़रूरी है)
>>> hf download hf://datasets/google/fleurs/data/

# पूरी repo डाउनलोड करें
>>> hf download hf://datasets/google/fleurs
```

CLI download command के बारे में विस्तृत जानकारी के लिए [CLI guide](./cli#hf-download) देखें।

## Dry-run mode

कभी-कभी actual डाउनलोड से पहले यह देखना ज़रूरी होता है कि कौन सी फ़ाइलें डाउनलोड होंगी। इसके लिए `--dry-run` parameter का उपयोग करें। यह repo की सभी डाउनलोड होने वाली फ़ाइलों की list दिखाता है और बताता है कि वे पहले से डाउनलोड हैं या नहीं। इससे यह समझ आता है कि कितनी फ़ाइलें और कितने size की डाउनलोड होनी हैं।

एकल फ़ाइल पर dry-run का उदाहरण:

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 1 files (out of 1) totalling 655.2M
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx 655.2M
```

और अगर फ़ाइल पहले से cached है:

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 0 files (out of 1) totalling 0.0.
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx -
```

पूरी repository पर भी dry-run किया जा सकता है:

```sh
>>> hf download openai-community/gpt2 --dry-run
[dry-run] Fetching 26 files: 100%|█████████████| 26/26 [00:04<00:00,  6.26it/s]
[dry-run] Will download 11 files (out of 26) totalling 5.6G.
File                              Bytes to download
--------------------------------- -----------------
.gitattributes                    -
64-8bits.tflite                   125.2M
64-fp16.tflite                    248.3M
64.tflite                         495.8M
README.md                         -
config.json                       -
flax_model.msgpack                497.8M
generation_config.json            -
merges.txt                        -
model.safetensors                 548.1M
onnx/config.json                  -
onnx/decoder_model.onnx           653.7M
onnx/decoder_model_merged.onnx    655.2M
onnx/decoder_with_past_model.onnx 653.7M
onnx/generation_config.json       -
onnx/merges.txt                   -
onnx/special_tokens_map.json      -
onnx/tokenizer.json               -
onnx/tokenizer_config.json        -
onnx/vocab.json                   -
pytorch_model.bin                 548.1M
rust_model.ot                     702.5M
tf_model.h5                       497.9M
tokenizer.json                    -
tokenizer_config.json             -
vocab.json                        -
```

और file filtering के साथ:

```sh
>>> hf download openai-community/gpt2 --include "*.json"  --dry-run
[dry-run] Fetching 11 files: 100%|█████████████| 11/11 [00:00<00:00, 80518.92it/s]
[dry-run] Will download 0 files (out of 11) totalling 0.0.
File                         Bytes to download
---------------------------- -----------------
config.json                  -
generation_config.json       -
onnx/config.json             -
onnx/generation_config.json  -
onnx/special_tokens_map.json -
onnx/tokenizer.json          -
onnx/tokenizer_config.json   -
onnx/vocab.json              -
tokenizer.json               -
tokenizer_config.json        -
vocab.json                   -
```

इसके अलावा, [`hf_hub_download`] और [`snapshot_download`] में `dry_run=True` pass करके programmatically भी dry-run किया जा सकता है। यह प्रत्येक फ़ाइल के लिए एक [`DryRunFileInfo`] (क्रमशः [`DryRunFileInfo`] की list) return करता है, जिसमें commit hash, file name, file size, फ़ाइल cached है या नहीं, और फ़ाइल डाउनलोड होगी या नहीं — यह सब जानकारी होती है। व्यवहार में, फ़ाइल तब डाउनलोड होगी जब वह cached न हो या `force_download=True` pass किया गया हो।

## तेज़ Downloads

`hf_xet` के ज़रिए तेज़ downloads का फायदा उठाएं। यह [`xet-core`](https://github.com/huggingface/xet-core) लाइब्रेरी का Python binding है जो तेज़ downloads और uploads के लिए chunk-based deduplication enable करता है। `hf_xet`, `huggingface_hub` के साथ seamlessly integrate होता है, लेकिन LFS की बजाय Rust `xet-core` लाइब्रेरी और Xet storage इस्तेमाल करता है।

`hf_xet`, Xet storage system का उपयोग करता है जो फ़ाइलों को immutable chunks में तोड़ता है, इन chunks के collections (जिन्हें blocks या xorbs कहते हैं) को remotely store करता है, और request पर फ़ाइल reassemble करने के लिए उन्हें retrieve करता है। डाउनलोड के दौरान, user का authorization confirm होने के बाद `hf_xet`, Xet content-addressable service (CAS) को फ़ाइल के LFS SHA256 hash के साथ query करता है — ताकि files assemble करने के लिए reconstruction metadata (xorbs के भीतर ranges) और xorbs को directly डाउनलोड करने के लिए presigned URLs मिल सकें। फिर `hf_xet` ज़रूरी xorb ranges efficiently डाउनलोड करके disk पर फ़ाइलें लिखता है।

इसे enable करने के लिए बस `huggingface_hub` का latest version install करें:

```bash
pip install -U "huggingface_hub"
```

`huggingface_hub` 0.32.0 से, इसके साथ `hf_xet` भी install होगा।

बाकी सभी `huggingface_hub` APIs बिना किसी बदलाव के काम करती रहेंगी। Xet storage और `hf_xet` के फायदों के बारे में अधिक जानने के लिए यह [section](https://huggingface.co/docs/hub/xet/index) देखें।

नोट: `hf_transfer` पहले LFS storage backend के साथ इस्तेमाल होता था और अब deprecated है — इसकी जगह `hf_xet` का उपयोग करें।