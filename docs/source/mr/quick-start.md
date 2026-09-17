<!--⚠️ लक्षात ठेवा की ही फाइल Markdown मध्ये आहे, परंतु यात आपल्या doc-builder साठीची विशेष syntax (MDX प्रमाणे) वापरली आहे. त्यामुळे ती तुमच्या Markdown viewer मध्ये योग्य प्रकारे render होईलच असे नाही.
-->

# त्वरित सुरुवात

[Hugging Face Hub](https://huggingface.co/) हे machine learning models, demos, datasets आणि metrics शेअर करण्यासाठी सर्वाधिक वापरले जाणारे प्लॅटफॉर्म आहे. `huggingface_hub` लायब्ररीमुळे तुमचे development environment न सोडता Hub सोबत सहज संवाद साधता येतो. याच्या मदतीने तुम्ही repositories सहज तयार आणि व्यवस्थापित करू शकता, files download आणि upload करू शकता, तसेच Hub वरून models आणि datasets ची उपयुक्त metadata मिळवू शकता.

## Installation

सुरुवात करण्यासाठी `huggingface_hub` लायब्ररी इंस्टॉल करा:

```bash
pip install --upgrade huggingface_hub
```

अधिक माहितीसाठी [installation](installation) मार्गदर्शिका पहा.

> [!TIP]
> `huggingface_hub` सोबत [`hf` CLI](./guides/cli) देखील उपलब्ध आहे, ज्यामुळे तुम्ही थेट terminal मधून Hub सोबत संवाद साधू शकता.
> जर तुम्ही AI agents (Claude Code, Codex, Cursor, ...) वापरत असाल, तर तुमच्या agent ला CLI वापरता यावे यासाठी Skill इंस्टॉल करा:
> ```bash
> # for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
> hf skills add
> # includes the above + Claude Code
> hf skills add --claude
> ```
> अधिक माहितीसाठी [Hugging Face CLI for AI Agents](https://huggingface.co/docs/hub/agents-cli) मार्गदर्शिका पहा.

## Files डाउनलोड करा

Hub वरील repositories या Git द्वारे version-controlled असतात. त्यामुळे वापरकर्ते एखादी स्वतंत्र file किंवा संपूर्ण repository डाउनलोड करू शकतात. Files डाउनलोड करण्यासाठी [`hf_hub_download`] function वापरू शकता. हे function file तुमच्या local disk वर डाउनलोड करून cache मध्ये साठवते. पुढील वेळी त्याच file ची आवश्यकता असल्यास ती थेट cache मधून वापरली जाईल, त्यामुळे पुन्हा डाउनलोड करण्याची गरज पडणार नाही.

File डाउनलोड करण्यासाठी तुम्हाला repository id आणि डाउनलोड करायच्या file चे नाव माहित असणे आवश्यक आहे. उदाहरणार्थ, [Pegasus](https://huggingface.co/google/pegasus-xsum) model ची configuration file डाउनलोड करण्यासाठी:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

File ची विशिष्ट version डाउनलोड करण्यासाठी `revision` parameter वापरून branch चे नाव, tag किंवा commit hash निर्दिष्ट करा. Commit hash वापरत असल्यास, 7-अक्षरी short hash ऐवजी पूर्ण (full-length) commit hash वापरणे आवश्यक आहे.

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

अधिक माहिती आणि उपलब्ध पर्यायांसाठी [`hf_hub_download`] चे API reference पहा.

<a id="login"></a> <!-- backward compatible anchor -->

## Authentication

अनेक प्रसंगी Hub शी संवाद साधण्यासाठी तुम्हाला Hugging Face account द्वारे authenticated असणे आवश्यक असते. उदाहरणार्थ: private repositories डाउनलोड करणे, files upload करणे, PRs तयार करणे इत्यादी.
तुमच्याकडे आधीपासून account नसेल, तर [नवीन account तयार करा](https://huggingface.co/join).

### Login command

Authenticate करण्याचा सर्वात सोपा मार्ग म्हणजे [`login`] command वापरणे:

```bash
hf auth login
```

जर तुम्ही आधीच login केले असेल, तर ही command लगेच पूर्ण होईल. पुन्हा login करण्यासाठी `hf auth login --force` वापरा. जर तुम्ही login केलेले नसाल, तर browser द्वारे login करण्यास सांगितले जाईल. त्यासाठी command मध्ये दाखवलेला URL उघडा, short code टाका, विनंतीला (request) मंजुरी द्या आणि access token मिळवून तो तुमच्या `HF_HOME` directory मध्ये जतन केला जाईल (डीफॉल्ट स्थान: `~/.cache/huggingface/token`).

हा token काही कालावधीनंतर expire होतो, परंतु तुम्ही त्याचा वापर करत राहिल्यास तो आपोआप refresh होतो. Hub शी संवाद साधणारी कोणतीही script किंवा library requests पाठवताना हा token वापरेल.

याशिवाय, तुम्ही तुमच्या [Settings page](https://huggingface.co/settings/tokens) वरून तयार केलेला [User Access Token](https://huggingface.co/docs/hub/security-tokens) देखील थेट वापरू शकता.

> [!TIP]
> User Access Tokens ना `read` किंवा `write` permissions असू शकतात. जर तुम्हाला repository तयार करायची किंवा त्यात बदल करायचे असतील, तर `write` access token वापरण्याची खात्री करा. अन्यथा, सुरक्षेच्या दृष्टीने `read` token तयार करणे अधिक योग्य आहे, कारण token चुकून उघड झाल्यास (leak झाल्यास) त्याचा धोका कमी होतो.

याशिवाय, notebook किंवा script मध्ये [`login`] वापरून programmatically देखील login करू शकता:

```py
>>> from huggingface_hub import login
>>> login()
```

एका वेळी तुम्ही फक्त एका account मध्येच login राहू शकता. नवीन account मध्ये login केल्यावर, आधीच्या account मधून आपोआप logout केले जाईल. सध्या कोणते account active आहे हे पाहण्यासाठी `hf auth whoami` command चालवा.

> [!WARNING]
> एकदा login केल्यानंतर, Hub कडे पाठवले जाणारे सर्व requests — अगदी authentication आवश्यक नसलेल्या methods सुद्धा — डीफॉल्टने तुमचा access token वापरतात. जर token चा हा implicit वापर बंद करायचा असेल, तर `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` हे environment variable सेट करा (अधिक माहितीसाठी [reference](../package_reference/environment_variables#hfhubdisableimplicittoken) पहा).

### अनेक tokens स्थानिकरित्या (locally) व्यवस्थापित करा

[`login`] command वापरून तुम्ही तुमच्या संगणकावर अनेक tokens जतन करू शकता. प्रत्येक token साठी एकदा login केल्यावर तो स्थानिकरित्या (locally) साठवला जातो. या tokens मध्ये बदल करण्यासाठी [`auth switch`] command वापरा:

```bash
hf auth switch
```

ही command जतन केलेल्या tokens ची यादी दाखवेल आणि तुम्हाला नावानुसार token निवडण्यास सांगेल. निवडल्यानंतर तो token _active_ token बनेल आणि Hub शी होणाऱ्या सर्व संवादांसाठी वापरला जाईल.

तुमच्या संगणकावर उपलब्ध असलेले सर्व access tokens पाहण्यासाठी `hf auth list` वापरा.

### Environment variable

Authentication साठी `HF_TOKEN` हे environment variable देखील वापरता येते. हे विशेषतः Hugging Face Space मध्ये उपयुक्त आहे, जिथे तुम्ही `HF_TOKEN` ला [Space secret](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) म्हणून सेट करू शकता.

> [!TIP]
> **नवीन:** Google Colaboratory मध्ये आता notebooks साठी [private keys](https://twitter.com/GoogleColab/status/1719798406195867814) define करता येतात. `HF_TOKEN` secret define केल्यास authentication आपोआप होईल!

Environment variable किंवा secret द्वारे केलेले authentication तुमच्या संगणकावर जतन केलेल्या token पेक्षा प्राधान्याने (priority) वापरले जाते.

### Method parameters

शेवटी, `token` parameter स्वीकारणाऱ्या कोणत्याही method मध्ये token थेट pass करूनही authentication करता येते.

```py
from huggingface_hub import whoami

user = whoami(token=...)
```

ही पद्धत सहसा वापरण्याची शिफारस केली जात नाही. मात्र, ज्या environment मध्ये token कायमस्वरूपी (permanently) साठवायचा नसेल किंवा एकाच वेळी अनेक tokens हाताळायचे असतील, अशा परिस्थितीत ती उपयुक्त ठरू शकते.

> [!WARNING]
> Token parameter म्हणून pass करताना काळजी घ्या. Token थेट codebase किंवा notebook मध्ये hardcode करण्याऐवजी, तो नेहमी secure vault मधून load करणे ही सर्वोत्तम पद्धत (best practice) आहे. Code मध्ये hardcoded tokens असल्यास, चुकून code share झाल्यास token leak होण्याचा मोठा धोका असतो.

## Repository तयार करा

Registration आणि login पूर्ण झाल्यानंतर, [`create_repo`] function वापरून repository तयार करा:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

जर repository private ठेवायची असेल, तर:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Private repositories फक्त तुम्हालाच दिसतील.

> [!TIP]
> Repository तयार करण्यासाठी किंवा Hub वर content push करण्यासाठी `write` permission असलेला User Access Token आवश्यक आहे. Token तयार करताना तुम्ही तुमच्या [Settings page](https://huggingface.co/settings/tokens) वरून ही permission निवडू शकता.

## Files upload करा

नवीन तयार केलेल्या repository मध्ये file जोडण्यासाठी [`upload_file`] function वापरा. यासाठी खालील माहिती द्यावी लागेल:

1. Upload करायच्या file चा path.
2. Repository मधील त्या file चा path.
3. ज्या repository मध्ये file जोडायची आहे त्या repository ची id.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

एकावेळी एकापेक्षा जास्त files upload करण्यासाठी [Upload](./guides/upload) मार्गदर्शिका पहा. त्यामध्ये Git वापरून किंवा Git शिवाय files upload करण्याच्या विविध पद्धतींची माहिती दिली आहे.

## पुढील दिशा

`huggingface_hub` लायब्ररीमुळे Python वापरून Hub शी सहज संवाद साधता येतो. Hub वरील files आणि repositories प्रभावीपणे कशा व्यवस्थापित करायच्या हे अधिक जाणून घेण्यासाठी, आमच्या [How-to Guides](./guides/overview) पाहण्याची आम्ही शिफारस करतो:

- [Repository व्यवस्थापित करा](./guides/repository).
- Hub वरून [Files डाउनलोड करा](./guides/download).
- Hub वर [Files upload करा](./guides/upload).
- तुम्हाला हवा असलेला model किंवा dataset शोधण्यासाठी [Hub मध्ये शोधा](./guides/search).
- Hugging Face Hub वर host केलेल्या models साठी विविध services वर [Inference चालवा](./guides/inference).