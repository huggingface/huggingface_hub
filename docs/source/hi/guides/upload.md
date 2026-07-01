<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hub पर फ़ाइलें अपलोड करें

अपनी फ़ाइलें और काम साझा करना Hub का एक महत्वपूर्ण पहलू है। `huggingface_hub` आपकी फ़ाइलों को Hub पर अपलोड करने के लिए कई विकल्प प्रदान करता है। आप इन functions का स्वतंत्र रूप से उपयोग कर सकते हैं या इन्हें अपनी library में एकीकृत कर सकते हैं, जिससे आपके users के लिए Hub के साथ इंटरैक्ट करना और भी सुविधाजनक हो जाता है।

जब भी आप Hub पर फ़ाइलें अपलोड करना चाहें, आपको अपने Hugging Face account में log in करना होगा। authentication के बारे में अधिक जानकारी के लिए, [इस अनुभाग](../quick-start#authentication) को देखें।

## एक फ़ाइल अपलोड करें

एक बार जब आप [`create_repo`] के साथ एक repository बना लेते हैं, तो आप [`upload_file`] का उपयोग करके अपनी repository में एक फ़ाइल अपलोड कर सकते हैं।

अपलोड करने वाली फ़ाइल का path, repository में आप उस फ़ाइल को कहाँ अपलोड करना चाहते हैं, और जिस repository में आप फ़ाइल जोड़ना चाहते हैं उसका नाम बताएं। अपनी repository type के आधार पर, आप वैकल्पिक रूप से repository type को `dataset`, `model`, या `space` के रूप में सेट कर सकते हैं।

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## एक फ़ोल्डर अपलोड करें

किसी मौजूदा repository में एक local फ़ोल्डर अपलोड करने के लिए [`upload_folder`] function का उपयोग करें। अपलोड करने वाले local फ़ोल्डर का path, repository में आप उस फ़ोल्डर को कहाँ अपलोड करना चाहते हैं, और जिस repository में आप फ़ोल्डर जोड़ना चाहते हैं उसका नाम बताएं। अपनी repository type के आधार पर, आप वैकल्पिक रूप से repository type को `dataset`, `model`, या `space` के रूप में सेट कर सकते हैं।

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# अपने remote Space पर local फ़ोल्डर की सारी सामग्री अपलोड करें।
# डिफ़ॉल्ट रूप से, फ़ाइलें repo के root पर अपलोड की जाती हैं
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

डिफ़ॉल्ट रूप से, यह जानने के लिए कि कौन सी फ़ाइलें commit की जानी चाहिए या नहीं, `.gitignore` फ़ाइल को ध्यान में रखा जाएगा। डिफ़ॉल्ट रूप से हम जाँचते हैं कि किसी commit में `.gitignore` फ़ाइल मौजूद है या नहीं, और अगर नहीं है, तो हम जाँचते हैं कि यह Hub पर मौजूद है या नहीं। कृपया ध्यान दें कि केवल directory के root पर मौजूद `.gitignore` फ़ाइल का ही उपयोग किया जाएगा। हम subdirectories में `.gitignore` फ़ाइलों की जाँच नहीं करते।

अगर आप hardcoded `.gitignore` फ़ाइल का उपयोग नहीं करना चाहते हैं, तो आप कौन सी फ़ाइलें अपलोड करनी हैं यह फ़िल्टर करने के लिए `allow_patterns` और `ignore_patterns` arguments का उपयोग कर सकते हैं। ये parameters या तो एक single pattern या patterns की एक list स्वीकार करते हैं। Patterns, Standard Wildcards (globbing patterns) होते हैं जैसा कि [यहाँ](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm) प्रलेखित है। अगर `allow_patterns` और `ignore_patterns` दोनों दिए गए हैं, तो दोनों constraints लागू होते हैं।

`.gitignore` फ़ाइल और allow/ignore patterns के अलावा, किसी भी subdirectory में मौजूद कोई भी `.git/` फ़ोल्डर अनदेखा कर दिया जाएगा।

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # किसी विशिष्ट फ़ोल्डर में अपलोड करें
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # सभी text logs को अनदेखा करें
... )
```

आप उसी commit में repo से जिन फ़ाइलों को delete करना चाहते हैं, उन्हें specify करने के लिए `delete_patterns` argument का भी उपयोग कर सकते हैं। यह तब उपयोगी हो सकता है जब आप किसी remote फ़ोल्डर में फ़ाइलें push करने से पहले उसे साफ़ करना चाहते हों और आपको यह न पता हो कि कौन सी फ़ाइलें पहले से मौजूद हैं।

नीचे दिया गया उदाहरण local `./logs` फ़ोल्डर को remote `/experiment/logs/` फ़ोल्डर में अपलोड करता है। केवल txt फ़ाइलें अपलोड की जाती हैं लेकिन उससे पहले, repo पर मौजूद सभी पुराने logs delete कर दिए जाते हैं। यह सब एक ही commit में।

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # सभी local text फ़ाइलें अपलोड करें
...     delete_patterns="*.txt", # पहले सभी remote text फ़ाइलें delete करें
... )
```

### फ़ाइलें कैसे अपलोड होती हैं

जब `hf_xet` इंस्टॉल होता है (जो कि डिफ़ॉल्ट रूप से होता है), तो [`upload_folder`] फ़ाइलों को एक streamed pipeline के माध्यम से अपलोड करता है: फ़ाइलों की Hub के विरुद्ध जाँच की जाती है, उन्हें Xet storage backend पर अपलोड किया जाता है (जो आंतरिक रूप से transfers को chunk, deduplicate और retry करता है), और adaptive batches में commit किया जाता है; यह सब समानांतर (parallel) रूप से होता है। व्यवहार में इसका मतलब है:

- **किसी भी आकार के फ़ोल्डर**: छोटे फ़ोल्डर एक ही commit में अपलोड हो जाते हैं, जबकि कई फ़ाइलों वाले फ़ोल्डर server limits से नीचे रहने के लिए स्वचालित रूप से कई commits में बाँट दिए जाते हैं। जब ऐसा होता है, तो बाद वाले commits के commit message पर एक ` (part 2)`, ` (part 3)`, ... suffix लग जाता है।
- **Resumable**: अगर किसी भी कारण से अपलोड बाधित हो जाता है, तो बस वही call दोबारा चलाएँ। पहले से commit की गई फ़ाइलों का पता लगाकर उन्हें skip कर दिया जाता है, और पहले से अपलोड किए गए chunks को deduplicate कर दिया जाता है — उन्हें दोबारा अपलोड करने पर (लगभग) कोई data transfer नहीं होता। इसमें कोई local state शामिल नहीं होता: आप किसी दूसरी machine से भी resume कर सकते हैं। एक अपवाद: `create_pr=True` के साथ, दोबारा चलाने पर एक नया pull request खुलता है। अपलोड resume करते समय हम इसके बजाय `revision="refs/pr/N"` के साथ दोबारा चलाने की सलाह देते हैं।
- **कोई double read नहीं**: फ़ाइलें अपलोड के लिए chunk किए जाने के दौरान ही, एक ही read pass में hash कर दी जाती हैं। अपलोड शुरू होने से पहले कोई अलग "hashing" phase नहीं होता।

एक live progress display तीनों चरणों पर नज़र रखता है:

```
Found 5,000 files to upload
  Preparing   ████████████████████  5,000 / 5,000 ✓
  Uploading   ██████████████░░░░░░  423 / 603 files  3.8GB · 19.7MB/s
  Committing  ██████████████████░░  4,580 / 5,000  6 commits
```

अगर `hf_xet` इंस्टॉल नहीं है, तो [`upload_folder`] legacy behavior पर वापस चला जाता है: पहले सब कुछ hash करें, HTTP के माध्यम से अपलोड करें, फिर एक ही commit बनाएँ। बेहतर robustness के लिए हम हमेशा `hf_xet` को इंस्टॉल रखने की सलाह देते हैं।

## CLI से अपलोड करें

आप Hub पर सीधे फ़ाइलें अपलोड करने के लिए terminal से `hf upload` command का उपयोग कर सकते हैं। आंतरिक रूप से यह ऊपर बताए गए वही [`upload_file`] और [`upload_folder`] helpers का उपयोग करता है।

आप या तो एक single फ़ाइल या एक पूरा फ़ोल्डर अपलोड कर सकते हैं:

```bash
# उपयोग:  hf upload [repo_id] [local_path] [path_in_repo]
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> hf upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` और `path_in_repo` वैकल्पिक हैं और इन्हें अंतर्निहित रूप से (implicitly) अनुमान लगाया जा सकता है। अगर `local_path` सेट नहीं है, तो tool जाँचेगा कि क्या किसी local फ़ोल्डर या फ़ाइल का नाम `repo_id` जैसा ही है। अगर ऐसा है, तो उसकी सामग्री अपलोड कर दी जाएगी। अन्यथा, एक exception raise होता है जो user से `local_path` को स्पष्ट रूप से सेट करने के लिए कहता है। किसी भी स्थिति में, अगर `path_in_repo` सेट नहीं है, तो फ़ाइलें repo के root पर अपलोड की जाती हैं।

destination को एक single `hf://` URI के रूप में भी व्यक्त किया जा सकता है जो `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]` grammar का पालन करता है (पूरे syntax के लिए [HF URIs reference](../package_reference/hf_uris) देखें)। repo type, revision और `path_in_repo` फिर URI से पढ़े जाते हैं, जिन्हें `--repo-type` और `--revision` options के साथ जोड़ा नहीं जा सकता:

```bash
# किसी विशिष्ट branch पर एक dataset में एक single फ़ाइल अपलोड करें
>>> hf upload hf://datasets/Wauplin/my-cool-dataset@my-branch/data/train.csv ./train.csv
https://huggingface.co/datasets/Wauplin/my-cool-dataset/blob/my-branch/data/train.csv
```

> [!TIP]
> बड़ी फ़ाइलों पर अधिकतम upload throughput के लिए, [`HF_XET_HIGH_PERFORMANCE=1`](../package_reference/environment_variables.md#hf_xet_high_performance) environment variable सेट करें। यह `hf_xet` का high-performance mode सक्षम करता है, जो उपलब्ध bandwidth और CPU cores का पूरा उपयोग करता है। ध्यान दें: legacy `HF_HUB_ENABLE_HF_TRANSFER=1` flag अब उपयोग नहीं होता क्योंकि `hf_transfer` को `hf_xet` के पक्ष में हटा दिया गया था — इसके बजाय `HF_XET_HIGH_PERFORMANCE=1` सेट करें।

CLI upload command के बारे में अधिक जानकारी के लिए, कृपया [CLI guide](./cli#hf-upload) देखें।

## एक बड़ा फ़ोल्डर अपलोड करें

[`upload_folder`] और `hf upload` command, Hub पर फ़ाइलें अपलोड करने के लिए go-to समाधान हैं, जिनमें बहुत बड़े फ़ोल्डर भी शामिल हैं। फ़ाइलें कई commits में Hub पर stream की जाती हैं और अगर प्रक्रिया बाधित हो जाती है तो यह स्वचालित रूप से resume हो जाती है। बस वही call दोबारा चलाएँ और पहले से अपलोड की गई फ़ाइलें skip कर दी जाती हैं।

```py
>>> api.upload_folder(
...     repo_id="HuggingFaceM4/Docmatix",
...     repo_type="dataset",
...     folder_path="/path/to/local/docmatix",
... )
```

या terminal से:

```sh
hf upload HuggingFaceM4/Docmatix --repo-type=dataset /path/to/local/docmatix
```

> [!WARNING]
> legacy [`upload_large_folder`] method और `hf upload-large-folder` command **deprecated** हैं और भविष्य के किसी release में हटा दिए जाएँगे। इसके बजाय [`upload_folder`] / `hf upload` का उपयोग करें।

### बड़े uploads के लिए टिप्स और ट्रिक्स

अपने repo में बड़ी मात्रा में data के साथ काम करते समय कुछ limitations का ध्यान रखना चाहिए। data को stream करने में लगने वाले समय को देखते हुए, प्रक्रिया के अंत में upload/push का fail हो जाना या degraded experience का सामना करना, चाहे वह hf.co पर हो या locally काम करते समय, बहुत परेशान करने वाला हो सकता है।

Hub पर अपनी repositories को कैसे structure करें, इसके best practices के लिए हमारी [Repository limitations and recommendations](https://huggingface.co/docs/hub/repositories-recommendations) guide देखें। चलिए कुछ व्यावहारिक tips के साथ आगे बढ़ते हैं ताकि आपकी upload प्रक्रिया यथासंभव सहज हो सके।

- **छोटे से शुरू करें**: हम आपकी upload script को test करने के लिए थोड़ी मात्रा में data के साथ शुरू करने की सलाह देते हैं। किसी script पर iterate करना तब आसान होता है जब fail होने में बहुत कम समय लगे।
- **failures की उम्मीद रखें**: बड़ी मात्रा में data stream करना चुनौतीपूर्ण है। आप नहीं जानते कि क्या हो सकता है, लेकिन यह मान लेना हमेशा बेहतर होता है कि कम से कम एक बार तो कुछ न कुछ fail होगा — चाहे वह आपकी machine, आपके connection, या हमारे servers की वजह से हो। उदाहरण के लिए, अगर आप बड़ी संख्या में फ़ाइलें अपलोड करने की योजना बना रहे हैं, तो अगला batch अपलोड करने से पहले locally यह ट्रैक रखना सबसे अच्छा है कि आपने कौन सी फ़ाइलें पहले ही अपलोड कर दी हैं। आपको इस बात का आश्वासन है कि पहले से commit की गई कोई LFS फ़ाइल कभी दो बार re-upload नहीं होगी लेकिन client-side पर इसे जाँचने से फिर भी कुछ समय बच सकता है। [`upload_folder`] आपके लिए यही करता है।
- **`hf_xet` का उपयोग करें**: यह Hub के लिए नए storage backend का लाभ उठाता है, Rust में लिखा गया है, और अब सभी के उपयोग के लिए उपलब्ध है। दरअसल, `huggingface_hub` का उपयोग करते समय `hf_xet` पहले से ही डिफ़ॉल्ट रूप से सक्षम होता है! अधिकतम performance के लिए, [`HF_XET_HIGH_PERFORMANCE=1`](../package_reference/environment_variables.md#hf_xet_high_performance) को एक environment variable के रूप में सेट करें। ध्यान रखें कि जब high performance mode सक्षम होता है, तो tool सभी उपलब्ध bandwidth और CPU cores का उपयोग करने का प्रयास करेगा।

## उन्नत सुविधाएँ

ज़्यादातर मामलों में, अपनी फ़ाइलों को Hub पर अपलोड करने के लिए आपको [`upload_file`] और [`upload_folder`] से ज़्यादा की ज़रूरत नहीं पड़ेगी।
हालाँकि, चीज़ों को आसान बनाने के लिए `huggingface_hub` में और भी उन्नत सुविधाएँ हैं। चलिए इन पर एक नज़र डालते हैं!

### तेज़ Uploads

`hf_xet` के माध्यम से तेज़ uploads का लाभ उठाएँ, जो [`xet-core`](https://github.com/huggingface/xet-core) library की Python binding है और तेज़ uploads व downloads के लिए chunk-based deduplication सक्षम करती है। `hf_xet` `huggingface_hub` के साथ सहजता से एकीकृत होता है, लेकिन LFS के बजाय Rust `xet-core` library और Xet storage का उपयोग करता है।

`hf_xet` Xet storage system का उपयोग करता है, जो फ़ाइलों को immutable chunks में तोड़ता है, इन chunks के संग्रह (जिन्हें blocks या xorbs कहा जाता है) को remotely संग्रहीत करता है और अनुरोध किए जाने पर फ़ाइल को फिर से जोड़ने के लिए उन्हें retrieve करता है। अपलोड करते समय, यह पुष्टि करने के बाद कि user को इस repo पर लिखने की अनुमति है, `hf_xet` फ़ाइलों को scan करेगा, उन्हें उनके chunks में तोड़ेगा और उन chunks को xorbs में एकत्र करेगा (और ज्ञात chunks में deduplicate करेगा), और फिर इन xorbs को Xet content-addressable service (CAS) पर अपलोड करेगा, जो xorbs की integrity सत्यापित करेगा, xorb metadata को LFS SHA256 hash के साथ register करेगा (lookup/download का समर्थन करने के लिए), और xorbs को remote storage पर लिखेगा।

इसे सक्षम करने के लिए, बस `huggingface_hub` का latest version इंस्टॉल करें:

```bash
pip install -U "huggingface_hub"
```

`huggingface_hub` 0.32.0 के अनुसार, यह `hf_xet` को भी इंस्टॉल करेगा।

अन्य सभी `huggingface_hub` APIs बिना किसी बदलाव के काम करती रहेंगी। Xet storage और `hf_xet` के लाभों के बारे में अधिक जानने के लिए, इस [section](https://huggingface.co/docs/hub/xet/index) को देखें।

**Cluster / Distributed Filesystem से Upload के लिए विचारणीय बातें**

किसी cluster से अपलोड करते समय, अपलोड की जा रही फ़ाइलें अक्सर किसी distributed या networked filesystem (NFS, EBS, Lustre, Fsx, आदि) पर रहती हैं। Xet storage उन फ़ाइलों को chunk करेगा और उन्हें locally blocks (जिन्हें xorbs भी कहा जाता है) में लिखेगा, और block पूरा होने पर उन्हें अपलोड करेगा। किसी distributed filesystem से अपलोड करते समय बेहतर performance के लिए, [`HF_XET_CACHE`](../package_reference/environment_variables#hfxetcache) को किसी ऐसी directory पर सेट करना सुनिश्चित करें जो किसी local disk पर हो (उदा. एक local NVMe या SSD disk)। Xet cache का डिफ़ॉल्ट स्थान `HF_HOME` के अंतर्गत (`~/.cache/huggingface/xet`) पर होता है और यह user की home directory में होने के कारण अक्सर distributed filesystem पर ही स्थित होता है।

### बिना रुकावट वाले uploads

कुछ मामलों में, आप अपने main thread को block किए बिना data push करना चाहते हैं। यह किसी training को जारी रखते हुए logs और artifacts अपलोड करने के लिए विशेष रूप से उपयोगी है। ऐसा करने के लिए, आप [`upload_file`] और [`upload_folder`] दोनों में `run_as_future` argument का उपयोग कर सकते हैं। यह एक [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) object लौटाएगा जिसका उपयोग आप upload की स्थिति जाँचने के लिए कर सकते हैं।

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # background में अपलोड करें (non-blocking action)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # upload के पूरा होने की प्रतीक्षा करें (blocking action)
...
```

> [!TIP]
> `run_as_future=True` का उपयोग करते समय background jobs को queue किया जाता है। इसका मतलब है कि
> आपको इस बात की गारंटी है कि jobs सही क्रम में execute होंगी।

भले ही background jobs ज़्यादातर data अपलोड करने/commits बनाने के लिए उपयोगी होती हैं, आप [`run_as_future`] का उपयोग करके अपनी पसंद की किसी भी method को queue कर सकते हैं। उदाहरण के लिए, आप इसका उपयोग एक repo बनाने और फिर background में उसमें data अपलोड करने के लिए कर सकते हैं। upload methods में built-in `run_as_future` argument इसके इर्द-गिर्द बस एक alias है।

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### repositories के बीच फ़ाइलें कॉपी करें

बड़े data को download या re-upload किए बिना Hub पर repositories के बीच फ़ाइलें या फ़ोल्डर कॉपी करने के लिए [`copy_files`] का उपयोग करें। यह तब उपयोगी होता है जब आप model variants में weights को duplicate करना चाहते हों, repos के बीच dataset फ़ाइलें कॉपी करना चाहते हों, या अपनी repositories में फ़ाइलों को पुनर्व्यवस्थित करना चाहते हों। पर्दे के पीछे, यह [`CommitOperationCopy`] operations के साथ एक commit बनाता है।

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# repos के बीच एक single फ़ाइल कॉपी करें
>>> api.copy_files(
...     "hf://username/source-model/weights.safetensors",
...     "hf://username/target-model/weights.safetensors",
... )

# एक पूरा फ़ोल्डर कॉपी करें
>>> api.copy_files(
...     "hf://datasets/username/source-dataset/data/",
...     "hf://datasets/username/target-dataset/data/",
... )
```

आप उसी repository के भीतर भी कॉपी कर सकते हैं:

```py
# उसी repo में एक फ़ाइल को duplicate करें
>>> api.copy_files(
...     "hf://username/my-model/config.json",
...     "hf://username/my-model/backup/config.json",
... )
```

> [!TIP]
> किसी फ़ोल्डर को कॉपी करते समय, source पर एक trailing `/` rsync-style semantics का उपयोग करता है, जिसका मतलब है कि फ़ोल्डर की *सामग्री* कॉपी की जाती है, बिना फ़ोल्डर को स्वयं nest किए। trailing `/` के बिना, फ़ोल्डर स्वयं destination पर nest हो जाता है।

> [!TIP]
> [`copy_files`] फ़ाइलों को [Buckets](./buckets) पर कॉपी करने का भी समर्थन करता है। अधिक जानकारी के लिए [Buckets guide](./buckets#copy-files-to-bucket) देखें।

### शेड्यूल किए गए uploads

Hugging Face Hub, data को save और version करना आसान बनाता है। हालाँकि, एक ही फ़ाइल को हज़ारों बार अपडेट करने में कुछ limitations हैं। उदाहरण के लिए, आप किसी training process के logs या किसी deployed Space पर user feedback को save करना चाह सकते हैं। ऐसे मामलों में, data को Hub पर एक dataset के रूप में अपलोड करना समझदारी है, लेकिन इसे ठीक से करना मुश्किल हो सकता है। मुख्य कारण यह है कि आप अपने data के हर अपडेट को version नहीं करना चाहते क्योंकि इससे git repository अनुपयोगी हो जाएगी। [`CommitScheduler`] class इस समस्या का एक समाधान प्रदान करती है।

विचार यह है कि एक background job चलाया जाए जो नियमित रूप से एक local फ़ोल्डर को Hub पर push करे। मान लीजिए आपके पास एक Gradio Space है जो input के रूप में कुछ text लेता है और उसके दो translations generate करता है। फिर, user अपना पसंदीदा translation चुन सकता है। हर run के लिए, आप परिणामों का विश्लेषण करने के लिए input, output, और user की पसंद को save करना चाहते हैं। यह [`CommitScheduler`] के लिए एक बिल्कुल सही use case है; आप data को Hub पर save करना चाहते हैं (संभावित रूप से लाखों user feedback), लेकिन आपको हर user के input को real-time में save करने की _ज़रूरत_ नहीं है। इसके बजाय, आप data को locally एक JSON फ़ाइल में save कर सकते हैं और उसे हर 10 मिनट में अपलोड कर सकते हैं। उदाहरण के लिए:

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# डेटा को save करने वाली फ़ाइल परिभाषित करें। पिछले run के मौजूदा data को overwrite न करने के लिए UUID का उपयोग करें।
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# नियमित uploads शेड्यूल करें। Remote repo और local फ़ोल्डर अगर पहले से मौजूद नहीं हैं तो बना दिए जाते हैं।
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# वह function परिभाषित करें जो user द्वारा अपना feedback submit करने पर call किया जाएगा (Gradio में call किया जाना है)
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Gradio शुरू करें
>>> with gr.Blocks() as demo:
>>>     ... # Gradio demo परिभाषित करें + `save_feedback` का उपयोग करें
>>> demo.launch()
```

और बस इतना ही! User input/outputs और feedback, Hub पर एक dataset के रूप में उपलब्ध होंगे। एक unique JSON फ़ाइल नाम का उपयोग करके, आपको इस बात की गारंटी है कि आप किसी पिछले run के data को या उसी repository में समवर्ती रूप से push करने वाले किसी अन्य Spaces/replicas के data को overwrite नहीं करेंगे।

[`CommitScheduler`] के बारे में अधिक जानकारी के लिए, यहाँ वह है जो आपको जानना चाहिए:
- **append-only:**
    यह माना जाता है कि आप फ़ोल्डर में केवल content जोड़ेंगे। आपको केवल मौजूदा फ़ाइलों में data append करना चाहिए या नई फ़ाइलें बनानी चाहिए। किसी फ़ाइल को delete या overwrite करने से आपकी repository corrupt हो सकती है।
- **git history**:
    scheduler फ़ोल्डर को हर `every` मिनट में commit करेगा। git repository को बहुत अधिक प्रदूषित करने से बचने के लिए, न्यूनतम 5 मिनट का मान सेट करने की सलाह दी जाती है। इसके अलावा, scheduler को empty commits से बचने के लिए डिज़ाइन किया गया है। अगर फ़ोल्डर में कोई नया content नहीं मिलता, तो निर्धारित commit को छोड़ दिया जाता है।
- **errors:**
    scheduler एक background thread के रूप में चलता है। यह तब शुरू होता है जब आप class को instantiate करते हैं और कभी नहीं रुकता। विशेष रूप से, अगर upload के दौरान कोई error होता है (उदाहरण: connection issue), तो scheduler उसे चुपचाप अनदेखा कर देगा और अगले निर्धारित commit पर फिर से प्रयास करेगा।
- **thread-safety:**
    ज़्यादातर मामलों में यह मान लेना सुरक्षित है कि आप किसी lock file की चिंता किए बिना किसी फ़ाइल में लिख सकते हैं। अगर आप upload के दौरान फ़ोल्डर में content लिखते हैं तो scheduler crash नहीं होगा या corrupt नहीं होगा। व्यवहार में, _यह संभव है_ कि heavy-loaded apps के लिए concurrency issues हों। इस स्थिति में, हम thread-safety सुनिश्चित करने के लिए `scheduler.lock` lock का उपयोग करने की सलाह देते हैं। lock केवल तभी block होता है जब scheduler बदलावों के लिए फ़ोल्डर को scan करता है, न कि जब यह data अपलोड करता है। आप सुरक्षित रूप से मान सकते हैं कि इससे आपके Space पर user experience प्रभावित नहीं होगा।

#### Space persistence डेमो

किसी Space से Hub पर एक Dataset में data को persist करना [`CommitScheduler`] का मुख्य use case है। use case के आधार पर, आप अपने data को अलग तरीके से structure करना चाह सकते हैं। structure को concurrent users और restarts के लिए robust होना चाहिए, जिसका मतलब अक्सर UUIDs generate करना होता है। robustness के अलावा, आपको बाद में पुन: उपयोग के लिए data को 🤗 Datasets library द्वारा पढ़ने योग्य format में अपलोड करना चाहिए। हमने एक [Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) बनाया है जो दिखाता है कि कई अलग-अलग data formats कैसे save करें (आपको इसे अपनी विशिष्ट ज़रूरतों के लिए adapt करना पड़ सकता है)।

#### कस्टम uploads

[`CommitScheduler`] यह मानता है कि आपका data append-only है और उसे "जैसा है वैसा" अपलोड किया जाना चाहिए। हालाँकि, आप data अपलोड होने के तरीके को customize करना चाह सकते हैं। आप ऐसा [`CommitScheduler`] से inherit करने वाली एक class बनाकर और `push_to_hub` method को overwrite करके कर सकते हैं (इसे जैसे चाहें वैसे overwrite करने के लिए स्वतंत्र हैं)। आपको इस बात की गारंटी है कि इसे एक background thread में हर `every` मिनट में call किया जाएगा। आपको concurrency और errors की चिंता करने की ज़रूरत नहीं है लेकिन आपको अन्य पहलुओं के बारे में सावधान रहना चाहिए, जैसे empty commits या duplicated data push करना।

नीचे दिए गए (सरलीकृत) उदाहरण में, हम Hub पर repo को overload होने से बचाने के लिए सभी PNG फ़ाइलों को एक single archive में zip करने के लिए `push_to_hub` को overwrite करते हैं:

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. PNG फ़ाइलें सूचीबद्ध करें
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # अगर commit करने के लिए कुछ नहीं है तो जल्दी return करें

        # 2. png फ़ाइलों को एक single archive में zip करें
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. archive अपलोड करें
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. बाद में re-uploading से बचने के लिए local png फ़ाइलें delete करें
        for png_file in png_files:
            png_file.unlink()
```

जब आप `push_to_hub` को overwrite करते हैं, तो आपके पास [`CommitScheduler`] के attributes तक पहुँच होती है और विशेष रूप से:
- [`HfApi`] client: `api`
- Folder parameters: `folder_path` और `path_in_repo`
- Repo parameters: `repo_id`, `repo_type`, `revision`
- Thread lock: `lock`

> [!TIP]
> custom schedulers के और उदाहरणों के लिए, हमारा [demo Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) देखें
> जिसमें आपके use cases के आधार पर अलग-अलग implementations हैं।

### create_commit

[`upload_file`] और [`upload_folder`] functions high-level APIs हैं जो आम तौर पर उपयोग में सुविधाजनक होते हैं। अगर आपको निचले स्तर पर काम करने की ज़रूरत नहीं है तो हम पहले इन functions को आज़माने की सलाह देते हैं। हालाँकि, अगर आप commit-level पर काम करना चाहते हैं, तो आप सीधे [`create_commit`] function का उपयोग कर सकते हैं।

[`create_commit`] द्वारा तीन प्रकार के operations समर्थित हैं:

- [`CommitOperationAdd`] Hub पर एक फ़ाइल अपलोड करता है। अगर फ़ाइल पहले से मौजूद है, तो फ़ाइल की सामग्री overwrite कर दी जाती है। यह operation दो arguments स्वीकार करता है:

  - `path_in_repo`: वह repository path जहाँ फ़ाइल अपलोड करनी है।
  - `path_or_fileobj`: या तो आपके filesystem पर किसी फ़ाइल का path या एक file-like object। यह Hub पर अपलोड करने वाली फ़ाइल की सामग्री है।

- [`CommitOperationDelete`] किसी repository से एक फ़ाइल या फ़ोल्डर हटाता है। यह operation `path_in_repo` को एक argument के रूप में स्वीकार करता है।

- [`CommitOperationCopy`] किसी repository के भीतर या repositories के बीच एक फ़ाइल कॉपी करता है। यह operation निम्नलिखित arguments स्वीकार करता है:

  - `src_path_in_repo`: कॉपी करने वाली फ़ाइल का repository path।
  - `path_in_repo`: वह repository path जहाँ फ़ाइल कॉपी की जानी चाहिए।
  - `src_revision`: वैकल्पिक - कॉपी करने वाली फ़ाइल की revision, अगर आप किसी अलग branch/revision से फ़ाइल कॉपी करना चाहते हैं।
  - `src_repo_id`: वैकल्पिक - जिस source repository से कॉपी करना है (उदा. `"username/source-model"`)। डिफ़ॉल्ट रूप से destination repository।
  - `src_repo_type`: वैकल्पिक - source repository का type (`"model"`, `"dataset"` या `"space"`)। `src_repo_id` सेट होने पर आवश्यक।

उदाहरण के लिए, अगर आप किसी Hub repository में दो फ़ाइलें अपलोड करना और एक फ़ाइल delete करना चाहते हैं:

1. किसी फ़ाइल को add या delete करने और किसी फ़ोल्डर को delete करने के लिए उपयुक्त `CommitOperation` का उपयोग करें:

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. अपने operations को [`create_commit`] में pass करें:

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

[`upload_file`] और [`upload_folder`] के अलावा, निम्नलिखित functions भी पर्दे के पीछे [`create_commit`] का उपयोग करते हैं:

- [`delete_file`] Hub पर किसी repository से एक single फ़ाइल delete करता है।
- [`delete_folder`] Hub पर किसी repository से एक पूरा फ़ोल्डर delete करता है।
- [`metadata_update`] किसी repository का metadata अपडेट करता है।

अधिक विस्तृत जानकारी के लिए, [`HfApi`] reference पर एक नज़र डालें।

### commit से पहले LFS फ़ाइलों को pre-upload करें

कुछ मामलों में, आप commit call करने से **पहले** S3 पर विशाल फ़ाइलें अपलोड करना चाह सकते हैं। उदाहरण के लिए, अगर आप किसी dataset को कई shards में commit कर रहे हैं जो in-memory generate होते हैं, तो out-of-memory issue से बचने के लिए आपको shards को एक-एक करके अपलोड करना होगा। एक समाधान यह है कि प्रत्येक shard को repo पर एक अलग commit के रूप में अपलोड किया जाए। पूरी तरह मान्य होते हुए भी, इस समाधान की खामी यह है कि यह दसियों commits generate करके git history को गड़बड़ कर सकता है। इस समस्या से निपटने के लिए, आप अपनी फ़ाइलों को एक-एक करके S3 पर अपलोड कर सकते हैं और फिर अंत में एक ही commit बना सकते हैं। यह [`preupload_lfs_files`] को [`create_commit`] के साथ मिलाकर उपयोग करने से संभव है।

> [!WARNING]
> यह एक power-user method है। फ़ाइलों को pre-upload करने के low-level logic को संभालने के बजाय सीधे [`upload_file`], [`upload_folder`] या [`create_commit`] का उपयोग करना ही
> अधिकांश मामलों में सही तरीका है। [`preupload_lfs_files`] की मुख्य चेतावनी यह है कि जब तक commit वास्तव में नहीं हो जाता, तब तक अपलोड की गई फ़ाइलें
> Hub पर repo पर पहुँच योग्य नहीं होतीं। अगर आपका कोई प्रश्न है, तो बेझिझक हमें हमारे Discord पर या किसी GitHub issue में ping करें।

यहाँ एक सरल उदाहरण है जो दर्शाता है कि फ़ाइलों को pre-upload कैसे करें:

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # generate होने वाले सभी `CommitOperationAdd` objects की list
>>> for i in range(5):
...     content = ... # binary content generate करें
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # commit बनाएँ
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

सबसे पहले, हम [`CommitOperationAdd`] objects को एक-एक करके बनाते हैं। एक वास्तविक उदाहरण में, इनमें generate किए गए shards होंगे। अगली फ़ाइल generate करने से पहले हर फ़ाइल अपलोड की जाती है। [`preupload_lfs_files`] step के दौरान, **`CommitOperationAdd` object mutate हो जाता है**। आपको इसका उपयोग केवल इसे सीधे [`create_commit`] में pass करने के लिए करना चाहिए। object का मुख्य अपडेट यह है कि **इसमें से binary content हटा दिया जाता है**, जिसका मतलब है कि अगर आप इसका कोई और reference संग्रहीत नहीं करते तो यह garbage-collect हो जाएगा। यह अपेक्षित है क्योंकि हम पहले से अपलोड की गई सामग्री को memory में नहीं रखना चाहते। अंत में हम सभी operations को [`create_commit`] में pass करके commit बनाते हैं। आप ऐसे अतिरिक्त operations (add, delete या copy) भी pass कर सकते हैं जो अभी तक process नहीं हुए हैं और उन्हें सही ढंग से संभाला जाएगा।
