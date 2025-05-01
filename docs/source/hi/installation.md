<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# स्थापना

आरंभ करने से पहले, आपको उपयुक्त पैकेज स्थापित करके अपना परिवेश सेटअप करना होगा।

`huggingface_hub` का परीक्षण **Python 3.8+** पर किया गया है।

## पिप के साथ स्थापित करें

[वर्चुअल वातावरण](https://docs.python.org/3/library/venv.html) में `huggingface_hub` इंस्टॉल करने की अत्यधिक अनुशंसा की जाती है।
यदि आप Python वर्चुअल वातावरण से अपरिचित हैं, तो इस [गाइड](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) पर एक नज़र डालें।
एक वर्चुअल वातावरण विभिन्न परियोजनाओं को प्रबंधित करना आसान बनाता है, और निर्भरताओं के बीच संगतता समस्याओं से बचाता है।

अपनी प्रोजेक्ट निर्देशिका में एक वर्चुअल वातावरण बनाकर प्रारंभ करें:

```bash
python -m venv .env
```

वर्चुअल वातावरण सक्रिय करें. Linux और macOS पर:

```bash
source .env/bin/activate
```

वर्चुअल वातावरण सक्रिय करें Windows पर:

```bash
.env/Scripts/activate
```

अब आप `huggingface_hub` [PyPi रजिस्ट्री से](https://pypi.org/project/huggingface-hub/), इंस्टॉल करने के लिए तैयार हैं:

```bash
pip install --upgrade huggingface_hub
```

एक बार हो जाने के बाद [चेक इंस्टालेशन](#चेक-इंस्टॉलेशन), यह सुनिश्चित करने के लिए कि वह ठीक से काम कर रहा है।

### वैकल्पिक निर्भरताएँ स्थापित करें

`huggingface_hub` की कुछ निर्भरताएं [वैकल्पिक](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) हैं क्योंकि उन्हें `huggingface_hub` की मुख्य विशेषताओं को चलाने की आवश्यकता नहीं है। हालाँकि, यदि वैकल्पिक निर्भरताएँ स्थापित नहीं हैं तो `huggingface_hub` की कुछ सुविधाएँ उपलब्ध नहीं हो सकती हैं।

आप `pip` के माध्यम से वैकल्पिक निर्भरताएँ स्थापित कर सकते हैं:
```bash
# Install dependencies for tensorflow-specific features
# /!\ Warning: this is not equivalent to `pip install tensorflow`
pip install 'huggingface_hub[tensorflow]'

# Install dependencies for both torch-specific and CLI-specific features.
pip install 'huggingface_hub[cli,torch]'
```

यहां `huggingface_hub` में वैकल्पिक निर्भरताओं की सूची दी गई है:
- `cli`: `huggingface_hub` के लिए अधिक सुविधाजनक CLI इंटरफ़ेस प्रदान करें।
- `fastai`, `torch`, `tensorflow`: फ्रेमवर्क-विशिष्ट सुविधाओं को चलाने के लिए निर्भरताएँ।
- `dev`: lib में योगदान करने के लिए निर्भरताएँ। इसमें 'परीक्षण' (परीक्षण चलाने के लिए), 'टाइपिंग' (टाइप चेकर चलाने के लिए) और 'गुणवत्ता' (लिंटर चलाने के लिए) शामिल हैं।


### स्रोत से इंस्टॉल करें

कुछ मामलों में, `huggingface_hub` को सीधे स्रोत से स्थापित करना दिलचस्प होता है।
यह आपको नवीनतम स्थिर संस्करण के बजाय अत्याधुनिक `main` संस्करण का उपयोग करने की अनुमति देता है।
`main` संस्करण नवीनतम विकास के साथ अद्यतित रहने के लिए उपयोगी है, उदाहरण के लिए यदि अंतिम आधिकारिक रिलीज के बाद से एक बग को ठीक किया गया है लेकिन अभी तक एक नई रिलीज शुरू नहीं की गई है।

हालांकि, इसका मतलब है कि `main` संस्करण हमेशा स्थिर नहीं हो सकता है।
हम `main` संस्करण को चालू रखने का प्रयास करते हैं, और अधिकांश समस्याएं आमतौर पर कुछ घंटों या एक दिन के भीतर हल हो जाती हैं।
यदि आप किसी समस्या का सामना करते हैं, तो कृपया एक समस्या खोलें ताकि हम इसे और भी जल्दी ठीक कर सकें!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

स्रोत से इंस्टॉल करते समय, आप एक विशिष्ट शाखा भी निर्दिष्ट कर सकते हैं।
यह तब उपयोगी होता है जब आप किसी नई सुविधा या नए बग-फिक्स का परीक्षण करना चाहते हैं जिसे अभी तक मर्ज नहीं किया गया है:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

एक बार हो जाने के बाद [चेक इंस्टालेशन](#चेक-इंस्टॉलेशन), यह सुनिश्चित करने के लिए कि वह ठीक से काम कर रहा है।

### संपादन योग्य इंस्टॉल

स्रोत से इंस्टॉल करने से आपको एक [संपादन योग्य इंस्टॉल](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) कर सकते हैं।
यदि आप `huggingface_hub` में योगदान करने की योजना बना रहे हैं और कोड में परिवर्तनों का परीक्षण करने की आवश्यकता है,
तो यह एक अधिक उन्नत इंस्टॉलेशन है।
आपको अपनी मशीन पर `huggingface_hub` की एक स्थानीय प्रति क्लोन करने की आवश्यकता है।

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

ये कमांड उस फ़ोल्डर को लिंक करेंगे जिसमें आपने रिपॉजिटरी को क्लोन किया था और आपके Python लाइब्रेरी पथ।
Python अब सामान्य लाइब्रेरी पथ के अलावा आपके द्वारा क्लोन किए गए फ़ोल्डर के अंदर भी देखेगा।
उदाहरण के लिए, यदि आपके Python पैकेज आमतौर पर `./.venv/lib/python3.11/site-packages/` में स्थापित होते हैं,
तो Python उस फ़ोल्डर को भी खोजेगा जिसे आपने `./huggingface_hub/` क्लोन किया था।

## कोंडा के साथ स्थापित करें

यदि आप इससे अधिक परिचित हैं, तो आप [conda-forge चैनल](https://anaconda.org/conda-forge/huggingface_hub) का उपयोग करके `huggingface_hub` इंस्टॉल कर सकते हैं:


```bash
conda install -c conda-forge huggingface_hub
```

एक बार हो जाने के बाद [चेक इंस्टालेशन](#चेक-इंस्टॉलेशन), यह सुनिश्चित करने के लिए कि वह ठीक से काम कर रहा है।

## स्थापना की जाँच करें

एक बार इंस्टॉल हो जाने पर, निम्नलिखित कमांड चलाकर जांचें कि `huggingface_hub` ठीक से काम करता है:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

यह कमांड हब से [gpt2](https://huggingface.co/gpt2) मॉडल के बारे में जानकारी प्राप्त करेगा।
आउटपुट इस तरह दिखना चाहिए:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## विंडोज़ सीमाएँ

हर जगह अच्छे एमएल को लोकतांत्रिक बनाने के हमारे लक्ष्य के साथ, हमने `huggingface_hub` को एक क्रॉस-प्लेटफ़ॉर्म लाइब्रेरी बनाने के लिए बनाया है
और विशेष रूप से यूनिक्स-आधारित और विंडोज सिस्टम दोनों पर सही ढंग से काम करने के लिए।
हालाँकि, ऐसे कुछ मामले हैं जहाँ विंडोज़ पर चलने पर `huggingface_hub` की कुछ सीमाएँ हैं।
यहां ज्ञात मुद्दों की एक विस्तृत सूची दी गई है। यदि आप [Github](https://github.com/huggingface/huggingface_hub/issues/new/choose) पर एक समस्या खोलकर किसी अनिर्दिष्ट समस्या का सामना करते हैं तो कृपया हमें बताएं।

- `huggingface_hub` का `cache` सिस्टम हब से डाउनलोड की गई फ़ाइलों को कुशलतापूर्वक `cache` करने के लिए सिमलिंक पर निर्भर करता है।
विंडोज़ पर, आपको सिमलिंक को सक्षम करने के लिए डेवलपर मोड को सक्रिय करना होगा या अपने स्क्रिप्ट को व्यवस्थापक के रूप में चलाना होगा।
यदि वे सक्रिय नहीं हैं, तो cache-सिस्टम अभी भी काम करता है लेकिन गैर-अनुकूलित तरीके से। अधिक जानकारी के लिए कृपया [cache सीमाएँ](./guides/manage-cache#limities) अनुभाग पढ़ें।
- हब पर फ़ाइलपथ में विशेष वर्ण हो सकते हैं (उदा. `"path/to?/my/file"`)।
विंडोज़ विशेष वर्णों पर अधिक प्रतिबंधात्मक है जिससे उन फ़ाइलों को विंडोज़ पर डाउनलोड करना असंभव हो जाता है।
उम्मीद है कि यह एक दुर्लभ मामला है। अगर आपको लगता है कि यह एक गलती है तो कृपया रेपो मालिक से संपर्क करें या समाधान निकालने के लिए हमसे संपर्क करें।


## अगले कदम

एक बार जब `huggingface_hub` आपकी मशीन पर ठीक से स्थापित हो जाता है,
तो आप आरंभ करने के लिए [पर्यावरण चर कॉन्फ़िगर करें](package_reference/environment_variables) या [हमारे गाइडों में से एक की जांच करें](guides/overview)।
