<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>Huggingface Hub के लिए आधिकारिक पायथन क्लाइंट।</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/ko/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_de.md">Deutsch</a> |
        <b>हिंदी</b>  |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文（简体）</a>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_kn.md">ಕನ್ನಡ</a>
    <p>
</h4>

---

**दस्तावेज़ीकरण**: <a href="https://hf.co/docs/huggingface_hub" target="_blank">https://hf.co/docs/huggingface_hub</a>

**सोर्स कोड**: <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub</a>

---

## huggingface_hub लाइब्रेरी में आपका स्वागत है

`huggingface_hub` लाइब्रेरी आपको [हगिंग फेस हब](https://huggingface.co/) के साथ बातचीत करने की अनुमति देती है, जो रचनाकारों और सहयोगियों के लिए ओपन-सोर्स मशीन लर्निंग का लोकतंत्रीकरण करने वाला एक मंच है। अपनी परियोजनाओं के लिए पूर्व-प्रशिक्षित मॉडल और डेटासेट खोजें या हब पर होस्ट किए गए हजारों मशीन लर्निंग ऐप्स के साथ खेलें। आप समुदाय के साथ अपने स्वयं के मॉडल, डेटासेट और डेमो भी बना और साझा कर सकते हैं। `huggingface_hub` लाइब्रेरी पायथन के साथ इन सभी चीजों को करने का एक आसान तरीका प्रदान करती है।

## प्रमुख विशेषताऐं

- [फ़ाइलें डाउनलोड करें](https://huggingface.co/docs/huggingface_hub/en/guides/download) हब से।
- [फ़ाइलें अपलोड करें](https://huggingface.co/docs/huggingface_hub/en/guides/upload) हब पर।
- [अपनी रिपॉजिटरी प्रबंधित करें](https://huggingface.co/docs/huggingface_hub/en/guides/repository)।
- तैनात मॉडलों पर [अनुमान चलाएँ](https://huggingface.co/docs/huggingface_hub/en/guides/inference)।
- मॉडल, डेटासेट और स्पेस के लिए [खोज](https://huggingface.co/docs/huggingface_hub/en/guides/search)।
- [मॉडल कार्ड साझा करें](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) अपने मॉडलों का दस्तावेजीकरण करने के लिए।
- [समुदाय के साथ जुड़ें](https://huggingface.co/docs/huggingface_hub/en/guides/community) पीआर और टिप्पणियों के माध्यम से।

## स्थापना

[pip](https://pypi.org/project/huggingface-hub/) के साथ `huggingface_hub` पैकेज इंस्टॉल करें:

```bash
pip install huggingface_hub
```

यदि आप चाहें, तो आप इसे [conda](https://huggingface.co/docs/huggingface_hub/en/installation#install-with-conda) से भी इंस्टॉल कर सकते हैं।

पैकेज को डिफ़ॉल्ट रूप से न्यूनतम रखने के लिए, `huggingface_hub` कुछ उपयोग मामलों के लिए उपयोगी वैकल्पिक निर्भरता के साथ आता है। उदाहरण के लिए, यदि आप अनुमान के लिए संपूर्ण अनुभव चाहते हैं, तो चलाएँ:

```bash
pip install huggingface_hub[inference]
```

अधिक इंस्टॉलेशन और वैकल्पिक निर्भरता जानने के लिए, [इंस्टॉलेशन गाइड](https://huggingface.co/docs/huggingface_hub/en/installation) देखें।

## जल्दी शुरू

### फ़ाइलें डाउनलोड करें

एकल फ़ाइल डाउनलोड करें

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

या एक संपूर्ण भंडार

```py
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

फ़ाइलें स्थानीय कैश फ़ोल्डर में डाउनलोड की जाएंगी. [this_guide] में अधिक विवरण (https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)।

### लॉग इन करें

Hugging Face Hub एप्लिकेशन को प्रमाणित करने के लिए टोकन का उपयोग करता है (देखें [docs](https://huggingface.co/docs/hub/security-tokens))। अपनी मशीन में लॉगिन करने के लिए, निम्नलिखित सीएलआई चलाएँ:

```bash
hf auth login
# या कृपया इसे एक पर्यावरण चर के रूप में निर्दिष्ट करें।
hf auth login --token $HUGGINGFACE_TOKEN
```

### एक रिपॉजिटरी बनाएं

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### फाइलें अपलोड करें

एकल फ़ाइल अपलोड करें

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

या एक संपूर्ण फ़ोल्डर

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

[अपलोड गाइड](https://huggingface.co/docs/huggingface_hub/en/guides/upload) में विवरण के लिए।

## हब से एकीकरण।

हम मुफ्त मॉडल होस्टिंग और वर्जनिंग प्रदान करने के लिए शानदार ओपन सोर्स एमएल लाइब्रेरीज़ के साथ साझेदारी कर रहे हैं। आप मौजूदा एकीकरण [यहां](https://huggingface.co/docs/hub/libraries) पा सकते हैं।

फायदे ये हैं:

- पुस्तकालयों और उनके उपयोगकर्ताओं के लिए निःशुल्क मॉडल या डेटासेट होस्टिंग।
- गिट-आधारित दृष्टिकोण के कारण, बहुत बड़ी फ़ाइलों के साथ भी अंतर्निहित फ़ाइल संस्करणिंग।
- सभी मॉडलों के लिए होस्टेड अनुमान एपीआई सार्वजनिक रूप से उपलब्ध है।
- अपलोड किए गए मॉडलों के साथ खेलने के लिए इन-ब्राउज़र विजेट।
- कोई भी आपकी लाइब्रेरी के लिए एक नया मॉडल अपलोड कर सकता है, उन्हें मॉडल को खोजने योग्य बनाने के लिए बस संबंधित टैग जोड़ना होगा।
- तेज़ डाउनलोड! हम डाउनलोड को जियो-रेप्लिकेट करने के लिए क्लाउडफ्रंट (एक सीडीएन) का उपयोग करते हैं ताकि वे दुनिया में कहीं से भी तेजी से चमक सकें।
- उपयोग आँकड़े और अधिक सुविधाएँ आने वाली हैं।

यदि आप अपनी लाइब्रेरी को एकीकृत करना चाहते हैं, तो चर्चा शुरू करने के लिए बेझिझक एक मुद्दा खोलें। हमने ❤️ के साथ एक [चरण-दर-चरण मार्गदर्शिका](https://huggingface.co/docs/hub/adding-a-library) लिखी, जिसमें दिखाया गया कि यह एकीकरण कैसे करना है।

## योगदान (सुविधा अनुरोध, बग, आदि) का अति स्वागत है 💙💚💛💜🧡❤️

योगदान के लिए हर किसी का स्वागत है और हम हर किसी के योगदान को महत्व देते हैं। कोड समुदाय की मदद करने का एकमात्र तरीका नहीं है।
प्रश्नों का उत्तर देना, दूसरों की मदद करना, उन तक पहुंचना और दस्तावेज़ों में सुधार करना समुदाय के लिए बेहद मूल्यवान है।
हमने संक्षेप में बताने के लिए एक [योगदान मार्गदर्शिका](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) लिखी है
इस भंडार में योगदान करने की शुरुआत कैसे करें।
