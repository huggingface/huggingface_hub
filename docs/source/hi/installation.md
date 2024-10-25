<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# स्थापना

आरंभ करने से पहले, आपको उपयुक्त पैकेज स्थापित करके अपना परिवेश सेटअप करना होगा।

`huggingface_hub` का परीक्षण **Python 3.8+** पर किया गया है।

## पिप के साथ स्थापित करें

[आभासी वातावरण] (https://docs.python.org/3/library/venv.html) में `huggingface_hub` इंस्टॉल करने की अत्यधिक अनुशंसा की जाती है।
यदि आप पायथन आभासी वातावरण से अपरिचित हैं, तो इस [गाइड] (https://package.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) पर एक नज़र डालें।
एक आभासी वातावरण विभिन्न परियोजनाओं को प्रबंधित करना आसान बनाता है, और निर्भरताओं के बीच संगतता समस्याओं से बचता है।

अपनी प्रोजेक्ट निर्देशिका में एक वर्चुअल वातावरण बनाकर प्रारंभ करें:

```bash
python -m venv .env
```

आभासी वातावरण सक्रिय करें. Linux और macOS पर:

```bash
source .env/bin/activate
```

आभासी वातावरण सक्रिय करें Windows पर:

```bash
.env/Scripts/activate
```

अब आप `huggingface_hub` [PyPi रजिस्ट्री से] (https://pypi.org/project/huggingface-hub/) इंस्टॉल करने के लिए तैयार हैं:

```bash
pip install --upgrade huggingface_hub
```

एक बार हो जाने पर, [चेक इंस्टालेशन](#चेक-इंस्टॉलेशन) सही ढंग से काम कर रहा है।

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

कुछ मामलों में, `huggingface_hub` को सीधे स्रोत से इंस्टॉल करना दिलचस्प है।
यह आपको नवीनतम स्थिर संस्करण के बजाय ब्लीडिंग एज 'मुख्य' संस्करण का उपयोग करने की अनुमति देता है।
उदाहरण के लिए, `मुख्य' संस्करण नवीनतम विकास के साथ अद्यतन रहने के लिए उपयोगी है
यदि पिछली आधिकारिक रिलीज़ के बाद से कोई बग ठीक कर दिया गया है लेकिन अभी तक कोई नई रिलीज़ जारी नहीं की गई है।

हालाँकि, इसका मतलब यह है कि 'मुख्य' संस्करण हमेशा स्थिर नहीं हो सकता है। हम इसे बनाए रखने का प्रयास करते हैं
`मुख्य' संस्करण चालू है, और अधिकांश समस्याएं आमतौर पर हल हो जाती हैं
कुछ घंटों या एक दिन के भीतर. यदि आपको कोई समस्या आती है, तो कृपया एक अंक खोलें ताकि हम ऐसा कर सकें
इसे और भी जल्दी ठीक करें!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

स्रोत से इंस्टॉल करते समय, आप एक विशिष्ट शाखा भी निर्दिष्ट कर सकते हैं। यह उपयोगी है यदि आप
किसी नई सुविधा या नए बग-फिक्स का परीक्षण करना चाहते हैं जिसे अभी तक मर्ज नहीं किया गया है:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

एक बार हो जाने पर, [चेक इंस्टालेशन](#चेक-इंस्टॉलेशन) सही ढंग से काम कर रहा है।

### संपादन योग्य इंस्टॉल

स्रोत से इंस्टॉल करने से आपको एक [संपादन योग्य इंस्टॉल](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) सेटअप करने की अनुमति मिलती है।
यदि आप `huggingace_hub` में योगदान देने की योजना बना रहे हैं तो यह अधिक उन्नत इंस्टॉलेशन है
और कोड में परिवर्तनों का परीक्षण करने की आवश्यकता है। आपको `huggingface_hub` की एक स्थानीय प्रति क्लोन करने की आवश्यकता है
आपकी मशीन पर.

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

ये कमांड उस फ़ोल्डर को लिंक करेंगे जिसे आपने रिपॉजिटरी में क्लोन किया है और आपके पायथन लाइब्रेरी पथ।
पाइथॉन अब सामान्य लाइब्रेरी पथों के अलावा आपके द्वारा क्लोन किए गए फ़ोल्डर के अंदर भी देखेगा।
उदाहरण के लिए, यदि आपके पायथन पैकेज आमतौर पर `./.venv/lib/python3.12/site-packages/` में स्थापित हैं,
पायथन आपके द्वारा क्लोन किए गए फ़ोल्डर `./huggingface_hub/` को भी खोजेगा।

## कोंडा के साथ स्थापित करें

यदि आप इससे अधिक परिचित हैं, तो आप [conda-forge चैनल](https://anaconda.org/conda-forge/huggingface_hub) का उपयोग करके `huggingface_hub` इंस्टॉल कर सकते हैं:


```bash
conda install -c conda-forge huggingface_hub
```

एक बार हो जाने पर, [चेक इंस्टालेशन](#चेक-इंस्टॉलेशन) सही ढंग से काम कर रहा है।

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

हर जगह अच्छे एमएल का लोकतंत्रीकरण करने के अपने लक्ष्य के साथ, हमने `huggingface_hub` का निर्माण किया
क्रॉस-प्लेटफ़ॉर्म लाइब्रेरी और विशेष रूप से यूनिक्स-आधारित और विंडोज़ दोनों पर सही ढंग से काम करने के लिए
सिस्टम. हालाँकि, ऐसे कुछ मामले हैं जहाँ `huggingface_hub` की कुछ सीमाएँ हैं
विंडोज़ पर चलाएँ. यहां ज्ञात मुद्दों की एक विस्तृत सूची दी गई है। कृपया हमें बताएं यदि आप
[जीथब पर एक मुद्दा] (https://github.com/huggingface/huggingface_hub/issues/new/choose) खोलकर किसी भी अज्ञात समस्या का सामना करें।

- `huggingface_hub` का कैश सिस्टम डाउनलोड की गई फ़ाइलों को कुशलतापूर्वक कैश करने के लिए सिम्लिंक पर निर्भर करता है
हब से. विंडोज़ पर, आपको डेवलपर मोड सक्रिय करना होगा या व्यवस्थापक के रूप में अपनी स्क्रिप्ट चलानी होगी
सिम्लिंक सक्षम करें. यदि वे सक्रिय नहीं हैं, तो कैश-सिस्टम अभी भी काम करता है लेकिन गैर-अनुकूलित तरीके से
ढंग। अधिक जानकारी के लिए कृपया [कैश सीमाएँ](./guides/manage-cache#limities) अनुभाग पढ़ें।
- हब पर फ़ाइलपथ में विशेष वर्ण हो सकते हैं (जैसे `"path/to?/my/file"`)। विंडोज़ है
[विशेष वर्णों] पर अधिक प्रतिबंधात्मक(https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)
जिससे विंडोज़ पर उन फ़ाइलों को डाउनलोड करना असंभव हो जाता है। उम्मीद है कि यह एक दुर्लभ मामला है.
यदि आपको लगता है कि यह कोई गलती है तो कृपया रेपो मालिक से संपर्क करें या इसका पता लगाने के लिए हमसे संपर्क करें
एक समाधान।


## अगले कदम

एक बार जब `huggingface_hub` आपकी मशीन पर ठीक से स्थापित हो जाए, तो आप चाहेंगे
आरंभ करने के लिए [पर्यावरण चर कॉन्फ़िगर करें] (पैकेज_संदर्भ/पर्यावरण_चर) या [हमारे गाइडों में से एक की जांच करें] (मार्गदर्शिकाएं/अवलोकन)।
