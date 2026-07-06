<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# एक repository बनाएँ और उसका प्रबंधन करें

Hugging Face Hub, git repositories का एक संग्रह है। [Git](https://git-scm.com/) software development में व्यापक रूप से उपयोग किया जाने वाला एक tool है जो सहयोगात्मक रूप से काम करते समय projects को आसानी से version करने में मदद करता है। यह मार्गदर्शिका आपको दिखाएगी कि Hub पर repositories के साथ कैसे इंटरैक्ट करें, विशेष रूप से:

- repository बनाएँ और delete करें।
- branches और tags का प्रबंधन करें।
- अपनी repository का नाम बदलें।
- अपनी repository की visibility अपडेट करें।
- अपनी repository की एक local copy का प्रबंधन करें।

> [!WARNING]
> अगर आप GitLab/GitHub/Bitbucket जैसे platforms के साथ काम करने के आदी हैं, तो आपकी पहली प्रवृत्ति
> अपने repo को clone करने (`git clone`), बदलावों को commit करने (`git add, git commit`) और उन्हें push करने (`git push`) के लिए `git` CLI का उपयोग करने की हो सकती है। Hugging Face Hub का उपयोग करते समय यह मान्य है। हालाँकि, software engineering और machine learning की
> requirements और workflows एक जैसी नहीं होतीं। Model repositories अलग-अलग frameworks और tools के लिए बड़ी model weight फ़ाइलें रख सकती हैं, इसलिए repository को clone करने से आपके पास बहुत बड़े local फ़ोल्डर्स हो सकते हैं।
> परिणामस्वरूप, हमारी custom HTTP methods का उपयोग करना अधिक कुशल हो सकता है। अधिक जानकारी के लिए आप हमारा [Git vs HTTP paradigm](../concepts/git_vs_http)
> व्याख्या पृष्ठ पढ़ सकते हैं।

अगर आप Hub पर एक repository बनाना और उसका प्रबंधन करना चाहते हैं, तो आपकी machine का logged in होना ज़रूरी है। अगर नहीं है, तो कृपया [इस अनुभाग](../quick-start#authentication) को देखें। इस मार्गदर्शिका के बाकी हिस्से में, हम यह मान लेंगे कि आपकी machine logged in है।

## अपनी repositories को सूचीबद्ध करें

आप [`list_user_repos`] का उपयोग करके अपने account या किसी organization की सभी repositories (models, datasets, spaces, और buckets) को सूचीबद्ध कर सकते हैं। परिणामों में storage जानकारी शामिल होती है और उन्हें storage usage के आधार पर sort किया जाता है।

```py
>>> from huggingface_hub import list_user_repos

# authenticated user के लिए repos सूचीबद्ध करें
>>> repos = list(list_user_repos())
>>> for repo in repos[:3]:
...     print(f"{repo.id} ({repo.type}) - {repo.storage} bytes")
username/my-model (model) - 4828692480 bytes
username/my-dataset (dataset) - 598427559 bytes
username/my-space (space) - 120620146 bytes

# किसी organization से repos सूचीबद्ध करें
>>> repos = list(list_user_repos(namespace="my-org"))
```

या CLI के माध्यम से (डिफ़ॉल्ट रूप से 30 repos दिखाता है, सभी को सूचीबद्ध करने के लिए `--limit 0` का उपयोग करें):

```bash
>>> hf repos ls
>>> hf repos ls --namespace my-org --type model
>>> hf repos ls --limit 0 --format json | jq '.[].id'
```

## Repo बनाना और delete करना

पहला कदम यह जानना है कि repositories कैसे बनाएँ और delete करें। आप केवल उन्हीं repositories का प्रबंधन कर सकते हैं जो आपके स्वामित्व में हैं (आपके username namespace के अंतर्गत) या उन organizations की जिनमें आपके पास write permissions हैं।

### एक repository बनाएँ

[`create_repo`] के साथ एक खाली repository बनाएँ और `repo_id` parameter के साथ उसे एक नाम दें। `repo_id` आपका namespace होता है जिसके बाद repository का नाम आता है: `username_or_org/repo_name`।

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

या CLI के माध्यम से:

```bash
>>> hf repos create lysandre/test-model
Successfully created lysandre/test-model on the Hub.
Your repo is now available at https://huggingface.co/lysandre/test-model
```

डिफ़ॉल्ट रूप से, [`create_repo`] एक model repository बनाता है। लेकिन आप किसी अन्य repository type को specify करने के लिए `repo_type` parameter का उपयोग कर सकते हैं। उदाहरण के लिए, अगर आप एक dataset repository बनाना चाहते हैं:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

या CLI के माध्यम से:

```bash
>>> hf repos create lysandre/test-dataset --repo-type dataset
```

जब आप एक repository बनाते हैं, तो आप `visibility` parameter के साथ अपनी repository की visibility सेट कर सकते हैं:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", visibility="private")
```

या CLI के माध्यम से:

```bash
>>> hf repos create lysandre/test-private --private
```

अगर आप बाद में repository की visibility बदलना चाहते हैं, तो आप [`update_repo_settings`] function का उपयोग कर सकते हैं।

> [!TIP]
> अगर आप Enterprise plan वाली किसी organization का हिस्सा हैं, तो आप [`create_repo`] को parameter के रूप में `resource_group_id` pass करके किसी विशिष्ट resource group में एक repo बना सकते हैं। Resource groups एक security feature हैं जो यह नियंत्रित करती हैं कि आपकी org के कौन से members किसी दिए गए resource तक पहुँच सकते हैं। आप resource group ID को Hub पर अपनी org settings पेज के URL से कॉपी करके प्राप्त कर सकते हैं (उदा. `"https://huggingface.co/organizations/huggingface/settings/resource-groups/66670e5163145ca562cb1988"` => `"66670e5163145ca562cb1988"`)। resource group के बारे में अधिक जानकारी के लिए, इस [guide](https://huggingface.co/docs/hub/en/security-resource-groups) को देखें।

आप `region` को parameter के रूप में pass करके किसी विशिष्ट cloud region में भी एक repo बना सकते हैं:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model", region="us")
```

### एक repository delete करें

[`delete_repo`] के साथ एक repository delete करें। सुनिश्चित करें कि आप वाकई एक repository delete करना चाहते हैं क्योंकि यह एक अपरिवर्तनीय (irreversible) प्रक्रिया है!

जिस repository को आप delete करना चाहते हैं उसका `repo_id` specify करें:

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

अगर repository मौजूद नहीं है तो call को चुपचाप अनदेखा करने के लिए `missing_ok=True` pass करें:

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset", missing_ok=True)
```

या CLI के माध्यम से:

```bash
>>> hf repos delete lysandre/my-corrupted-dataset --repo-type dataset
```

### एक repository को duplicate करें

कुछ मामलों में, आप किसी और के repo को अपने use case के अनुसार adapt करने के लिए कॉपी करना चाहते हैं।
यह [`duplicate_repo`] method का उपयोग करके संभव है। यह पूरी repository को duplicate करेगा, और पूरी git history को संरक्षित रखेगा।
यह models, datasets, और Spaces के लिए काम करता है। Spaces के लिए, आपको फिर भी अपनी खुद की settings (hardware, sleep-time, storage, variables और secrets) configure करनी होंगी। अधिक जानकारी के लिए हमारी [Manage your Space](./manage-spaces) मार्गदर्शिका देखें।

```py
>>> from huggingface_hub import duplicate_repo

# एक Space को duplicate करें
>>> duplicate_repo("multimodalart/dreambooth-training", repo_type="space", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)

# एक dataset को duplicate करें
>>> duplicate_repo("openai/gdpval", repo_type="dataset")
RepoUrl('https://huggingface.co/datasets/nateraw/gdpval',...)
```

## Spaces खोजें

Hub, Spaces खोजने के लिए एक semantic search API प्रदान करता है। आप [`search_spaces`] के साथ natural language queries का उपयोग करके खोज सकते हैं:

```py
>>> from huggingface_hub import search_spaces
>>> results = list(search_spaces("generate image"))
>>> results[0].id
'mrfakename/Z-Image-Turbo'
```

अधिक जानकारी और filtering options के लिए, [Manage your Spaces](./manage-spaces#search-for-spaces) मार्गदर्शिका देखें।

## फ़ाइलें अपलोड और डाउनलोड करें

अब जब आपने अपनी repository बना ली है, तो आप उसमें बदलाव push करने और उससे फ़ाइलें डाउनलोड करने में रुचि रखते हैं।

इन 2 विषयों के लिए अलग मार्गदर्शिकाएँ उपलब्ध हैं। अपनी repository का उपयोग कैसे करें यह सीखने के लिए कृपया [upload](./upload) और [download](./download) मार्गदर्शिकाएँ देखें।

## फ़ाइलें कॉपी करें

Hub पर पहले से host की गई फ़ाइलों को एक repository से दूसरी में (या यहाँ तक कि उसी repository के भीतर) बिना download और re-upload किए कॉपी करने के लिए [`copy_files`] का उपयोग करें। अलग-अलग फ़ाइलें और पूरे फ़ोल्डर दोनों समर्थित हैं, और Xet या LFS के साथ track की गई फ़ाइलें hash द्वारा server-side कॉपी की जाती हैं।

```py
>>> from huggingface_hub import copy_files

# एक repo से दूसरे repo में एक single फ़ाइल कॉपी करें
>>> copy_files(
...     "hf://username/source-model/config.json",
...     "hf://username/dest-model/config.json",
... )

# एक पूरा फ़ोल्डर कॉपी करें (trailing "/" फ़ोल्डर की *सामग्री* कॉपी करता है, rsync-style)
>>> copy_files(
...     "hf://datasets/username/my-dataset/data/",
...     "hf://datasets/username/my-dataset-copy/data/",
... )
```

या CLI के माध्यम से, unified `hf cp` command के साथ (जो `hf repos cp` के रूप में भी उपलब्ध है):

```bash
# repositories के बीच एक single फ़ाइल कॉपी करें
>>> hf cp hf://username/source-model/config.json hf://username/dest-model/config.json

# किसी repo से अपनी local machine पर एक फ़ाइल कॉपी करें
>>> hf repos cp hf://username/my-model/config.json ./config.json

# किसी repository पर एक local फ़ाइल अपलोड करें
>>> hf repos cp ./model.safetensors hf://username/my-model/model.safetensors
```

> [!TIP]
> `copy_files` (और `hf cp`) किसी repository से किसी [Bucket](./buckets) में भी फ़ाइलें कॉपी कर सकते हैं। किसी bucket *से* किसी repository *में* कॉपी करना समर्थित नहीं है। अधिक जानकारी के लिए [Buckets](./buckets) मार्गदर्शिका देखें।

> [!WARNING]
> Server-side copies केवल एक ही [storage region](https://huggingface.co/docs/hub/storage-regions) के भीतर काम करती हैं।

## Branches और tags

Git repositories अक्सर एक ही repository के अलग-अलग versions संग्रहीत करने के लिए branches का उपयोग करती हैं।
Tags का उपयोग आपकी repository की किसी विशिष्ट स्थिति को flag करने के लिए भी किया जा सकता है, उदाहरण के लिए, जब कोई version release किया जाता है।
अधिक सामान्य रूप से, branches और tags को [git references](https://git-scm.com/book/en/v2/Git-Internals-Git-References) कहा जाता है।

### Branches और tags बनाएँ

आप [`create_branch`] और [`create_tag`] का उपयोग करके नई branch और tags बना सकते हैं:

```py
>>> from huggingface_hub import create_branch, create_tag

# `main` branch से किसी Space repo पर एक branch बनाएँ
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# `v0.1-release` branch से किसी Dataset repo पर एक tag बनाएँ
>>> create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

या CLI के माध्यम से:

```bash
>>> hf repos branch create Matthijs/speecht5-tts-demo handle-dog-speaker --repo-type space
>>> hf repos tag create bigcode/the-stack v0.1.1 --repo-type dataset --revision v0.1-release -m "Bump release version."
```

किसी branch या tag को delete करने के लिए आप उसी तरह [`delete_branch`] और [`delete_tag`] functions का उपयोग कर सकते हैं, या CLI में क्रमशः `hf repos branch delete` और `hf repos tag delete` का।

### सभी branches और tags सूचीबद्ध करें

आप [`list_repo_refs`] का उपयोग करके किसी repository से मौजूदा git refs भी सूचीबद्ध कर सकते हैं:

```py
>>> from huggingface_hub import list_repo_refs
>>> list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## Repository settings बदलें

Repositories कुछ settings के साथ आती हैं जिन्हें आप configure कर सकते हैं। अधिकांश समय, आप ऐसा अपने browser में repo settings पेज में manually करना चाहेंगे। किसी repo को configure करने के लिए आपके पास उस पर write access होना ज़रूरी है (या तो उसके मालिक हों या किसी organization का हिस्सा हों)। इस अनुभाग में, हम उन settings को देखेंगे जिन्हें आप `huggingface_hub` का उपयोग करके programmatically भी configure कर सकते हैं।

कुछ settings Spaces के लिए विशिष्ट होती हैं (hardware, environment variables,...)। उन्हें configure करने के लिए, कृपया हमारी [Manage your Spaces](../guides/manage-spaces) मार्गदर्शिका देखें।

### Visibility अपडेट करें

एक repository public या private हो सकती है। एक private repository केवल आपको या उस organization के members को दिखाई देती है जिसमें repository स्थित है। किसी repository को private में बदलें जैसा कि नीचे दिखाया गया है:

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

या CLI के माध्यम से:

```bash
>>> hf repos settings lysandre/test-private --private true
```

### Gated access सेट करें

repos का उपयोग कैसे किया जाता है इस पर अधिक नियंत्रण देने के लिए, Hub, repo authors को अपने repos के लिए **access requests** सक्षम करने की अनुमति देता है। सक्षम होने पर, फ़ाइलों तक पहुँचने के लिए user को repo authors के साथ अपनी contact जानकारी (username और email address) साझा करने के लिए सहमत होना होगा। access requests सक्षम वाले repo को **gated repo** कहा जाता है।

आप [`update_repo_settings`] का उपयोग करके किसी repo को gated के रूप में सेट कर सकते हैं:

```py
>>> from huggingface_hub import HfApi

>>> api = HfApi()
>>> api.update_repo_settings(repo_id=repo_id, gated="auto")  # किसी model के लिए automatic gating सेट करें
```

या CLI के माध्यम से:

```bash
>>> hf repos settings lysandre/test-private --gated auto
```

### अपनी repository का नाम बदलें

आप [`move_repo`] का उपयोग करके Hub पर अपनी repository का नाम बदल सकते हैं। इस method का उपयोग करके, आप repo को किसी user से किसी organization में भी move कर सकते हैं। ऐसा करते समय, कुछ [सीमाएँ](https://hf.cos/docs/hub/repositories-settings#renaming-or-transferring-a-repo) हैं जिनके बारे में आपको पता होना चाहिए। उदाहरण के लिए, आप अपने repo को किसी अन्य user को transfer नहीं कर सकते।

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

या CLI के माध्यम से:

```bash
>>> hf repos move Wauplin/cool-model huggingface/cool-model
```

## Kernel repositories

Hub, compute kernels को host करने के लिए एक `"kernel"` repository type का समर्थन करता है। यह एक पूरी तरह से compatible repo type **नहीं** है। केवल methods का एक सीमित सेट ही test किया गया है और आधिकारिक रूप से समर्थित है:

- [`kernel_info`]
- [`hf_hub_download`]
- [`snapshot_download`]
- [`list_repo_refs`]
- [`list_repo_files`]
- [`list_repo_tree`]

ध्यान दें कि [`create_repo`] और [`delete_repo`] भी compatible हैं लेकिन Hub पर अनुमत users और orgs के एक छोटे subset तक सीमित हैं।

kernel repos को build, publish, और उपयोग करने के लिए, कृपया इसके बजाय समर्पित [`kernels`](https://github.com/huggingface/kernels) package का उपयोग करें। अधिक जानकारी के लिए [Kernels documentation](https://huggingface.co/docs/kernels/index) देखें।
