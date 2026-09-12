<!--⚠️ लक्षात ठेवा की ही फाइल Markdown मध्ये आहे, परंतु यात आपल्या doc-builder साठीची विशेष syntax (MDX प्रमाणे) वापरली आहे. त्यामुळे ती तुमच्या Markdown viewer मध्ये योग्य प्रकारे render होईलच असे नाही.
-->

# प्रतिष्ठापन

सुरुवात करण्यापूर्वी, आवश्यक packages इंस्टॉल करून तुमचे environment तयार करणे गरजेचे आहे.

`huggingface_hub` ची चाचणी **Python 3.10+** वर करण्यात आली आहे.

## pip वापरून इंस्टॉल करा

`huggingface_hub` [virtual environment](https://docs.python.org/3/library/venv.html) मध्ये इंस्टॉल करण्याची जोरदार शिफारस केली जाते.
जर तुम्हाला Python virtual environments बद्दल माहिती नसेल, तर हे [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) पहा.
Virtual environment मुळे वेगवेगळे प्रकल्प व्यवस्थापित करणे सोपे होते आणि dependencies मधील compatibility संबंधित समस्या टाळण्यास मदत होते.

सुरुवातीला, तुमच्या project directory मध्ये virtual environment तयार करा:

```bash
python -m venv .venv
```

Linux आणि macOS वर virtual environment सक्रिय करा:

```bash
source .venv/bin/activate
```

Windows वर virtual environment सक्रिय करा:

```bash
.venv/Scripts/activate
```

आता तुम्ही `huggingface_hub` ला [PyPI registry](https://pypi.org/project/huggingface-hub/) मधून इंस्टॉल करण्यासाठी तयार आहात:

```bash
pip install --upgrade huggingface_hub
```

इंस्टॉलेशन पूर्ण झाल्यानंतर, सर्व काही योग्यरित्या कार्य करत आहे याची खात्री करण्यासाठी [installation तपासा](#check-installation).

### पर्यायी dependencies इंस्टॉल करा

`huggingface_hub` मधील काही dependencies [optional](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) आहेत, कारण `huggingface_hub` ची मुख्य वैशिष्ट्ये (core features) वापरण्यासाठी त्यांची आवश्यकता नसते. मात्र, या optional dependencies इंस्टॉल केल्या नसल्यास `huggingface_hub` मधील काही वैशिष्ट्ये उपलब्ध नसू शकतात.

तुम्ही `pip` वापरून optional dependencies इंस्टॉल करू शकता:

```bash
# Install dependencies for both torch-specific and MCP-specific features.
pip install 'huggingface_hub[mcp,torch]'
```

`huggingface_hub` मधील optional dependencies ची यादी खालीलप्रमाणे आहे:

- `fastai`, `torch`: framework-specific वैशिष्ट्ये वापरण्यासाठी आवश्यक dependencies.
- `dev`: लायब्ररीमध्ये योगदान देण्यासाठी आवश्यक dependencies. यात `testing` (tests चालवण्यासाठी), `typing` (type checker चालवण्यासाठी) आणि `quality` (linters चालवण्यासाठी) यांचा समावेश आहे.

### Source मधून इंस्टॉल करा

काही परिस्थितींमध्ये `huggingface_hub` थेट source मधून इंस्टॉल करणे उपयुक्त ठरू शकते.
यामुळे तुम्ही नवीनतम stable version ऐवजी अत्याधुनिक `main` version वापरू शकता.
उदाहरणार्थ, मागील अधिकृत release नंतर एखादा bug दुरुस्त झाला असेल, पण त्यासाठी नवीन release अद्याप प्रकाशित झाला नसेल, तर `main` version वापरून तुम्ही त्या नवीन बदलांचा लाभ घेऊ शकता.

मात्र, याचा अर्थ `main` version नेहमीच पूर्णपणे स्थिर (stable) असेलच असे नाही.
आम्ही `main` version कार्यरत ठेवण्याचा सातत्याने प्रयत्न करतो आणि बहुतेक समस्या काही तासांत किंवा एका दिवसाच्या आत सोडवल्या जातात.
तुम्हाला कोणतीही समस्या आढळल्यास, कृपया एक Issue उघडा, म्हणजे आम्ही ती आणखी लवकर दुरुस्त करू शकू!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Source मधून install करताना, तुम्ही विशिष्ट branch देखील निवडू शकता.
जर तुम्हाला अद्याप merge न झालेल्या नवीन feature किंवा bug fix ची चाचणी करायची असेल, तर हे उपयुक्त ठरते.

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

इंस्टॉलेशन पूर्ण झाल्यानंतर, सर्व काही योग्यरित्या कार्य करत आहे याची खात्री करण्यासाठी [installation तपासा](#check-installation).

### Editable install

Source मधून इंस्टॉल केल्यावर तुम्ही [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) देखील सेट अप करू शकता.
जर तुम्ही `huggingface_hub` मध्ये योगदान देण्याचा विचार करत असाल आणि कोडमधील बदलांची चाचणी करायची असेल, तर ही अधिक प्रगत (advanced) इंस्टॉलेशन पद्धत उपयुक्त आहे.
यासाठी तुम्हाला तुमच्या संगणकावर `huggingface_hub` ची local copy clone करावी लागेल.

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

या commands मुळे तुम्ही clone केलेले repository folder आणि Python library paths एकमेकांशी लिंक केले जातात.
यानंतर Python नेहमीच्या library paths बरोबरच तुम्ही clone केलेल्या folder मधूनही packages शोधेल.
उदाहरणार्थ, जर तुमची Python packages साधारणपणे `./.venv/lib/python3.13/site-packages/` मध्ये इंस्टॉल होत असतील, तर Python `./huggingface_hub/` या clone केलेल्या folder मधूनही शोध घेईल.

## Hugging Face CLI इंस्टॉल करा

तुमच्या Python environment मध्ये कोणतेही बदल न करता `hf` CLI सेट अप करण्यासाठी आमच्या one-liner installers चा वापर करा.

macOS आणि Linux वर:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Windows वर:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

आधीपासून इंस्टॉल केलेले version अपडेट करण्यासाठी `hf update` चालवा. हे `hf` कसे इंस्टॉल केले गेले आहे (standalone installer, Homebrew किंवा pip) हे आपोआप ओळखते आणि त्यानुसार योग्य command चालवते.

## conda वापरून इंस्टॉल करा

जर तुम्ही `conda` वापरत असाल, तर [conda-forge channel](https://anaconda.org/conda-forge/huggingface_hub) वापरून `huggingface_hub` इंस्टॉल करू शकता.

```bash
conda install -c conda-forge huggingface_hub
```

इंस्टॉलेशन पूर्ण झाल्यानंतर, सर्व काही योग्यरित्या कार्य करत आहे याची खात्री करण्यासाठी [installation तपासा](#check-installation).

## Installation तपासा

इंस्टॉल केल्यानंतर, खालील command चालवून `huggingface_hub` योग्यरित्या कार्य करत आहे का ते तपासा:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

ही command Hub वरून [gpt2](https://huggingface.co/gpt2) मॉडेलची माहिती मिळवेल.
आउटपुट साधारणपणे खालीलप्रमाणे दिसेल:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows वरील मर्यादा

सर्वांसाठी मशीन लर्निंग अधिक सुलभ करण्याच्या आमच्या उद्दिष्टानुसार, `huggingface_hub` ही cross-platform लायब्ररी म्हणून विकसित करण्यात आली आहे, जेणेकरून ती Unix-आधारित आणि Windows या दोन्ही प्रणालींवर योग्यरित्या कार्य करेल. तरीही, Windows वर `huggingface_hub` वापरताना काही मर्यादा आहेत. खाली सध्या ज्ञात असलेल्या सर्व मर्यादांची यादी दिली आहे. याशिवाय तुम्हाला एखादी समस्या आढळल्यास, कृपया [GitHub वर issue उघडून](https://github.com/huggingface/huggingface_hub/issues/new/choose) आम्हाला कळवा.

- `huggingface_hub` ची cache system Hub वरून डाउनलोड केलेल्या files कार्यक्षमतेने cache करण्यासाठी **symlinks** वर अवलंबून असते. Windows वर symlinks सक्षम करण्यासाठी तुम्हाला Developer Mode सुरू करावा लागेल किंवा तुमची script administrator म्हणून चालवावी लागेल. Symlinks सक्षम नसले तरी cache system कार्यरत राहते, परंतु ती पूर्णपणे अनुकूल (optimized) पद्धतीने कार्य करत नाही. अधिक माहितीसाठी [cache limitations](./guides/manage-cache#limitations) विभाग पहा.
- Hub वरील file paths मध्ये विशेष अक्षरे (उदा. `"path/to?/my/file"`) असू शकतात. Windows मध्ये [special characters](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names) संदर्भात अधिक निर्बंध असल्यामुळे अशा files Windows वर डाउनलोड करणे शक्य नसते. ही परिस्थिती क्वचितच उद्भवते. तुम्हाला वाटत असेल की ही चूक आहे, तर कृपया त्या repository च्या मालकाशी संपर्क साधा किंवा योग्य उपाय शोधण्यासाठी आमच्याशी संपर्क साधा.

## पुढील पायऱ्या

तुमच्या संगणकावर `huggingface_hub` यशस्वीरित्या इंस्टॉल झाल्यानंतर, सुरुवात करण्यासाठी तुम्ही [environment variables configure करा](package_reference/environment_variables) किंवा आमच्या [guides पैकी एखादी पहा](guides/overview).