<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Model Cards बनाएँ और शेयर करें

`huggingface_hub` library Model Cards को बनाने, शेयर करने और अपडेट करने के लिए एक Python interface देती है।
Hub पर Model Cards क्या होते हैं और अंदर से कैसे काम करते हैं, यह विस्तार से जानने के लिए
[इस समर्पित documentation पेज](https://huggingface.co/docs/hub/models-cards) को देखें।

## Hub से Model Card लोड करें

Hub से किसी मौजूदा card को लोड करने के लिए आप [`ModelCard.load`] function का उपयोग कर सकते हैं। यहाँ हम [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans) का card लोड करेंगे।

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

इस card में कुछ उपयोगी attributes हैं जिन्हें आप एक्सेस करना या इस्तेमाल करना चाहेंगे:
  - `card.data`: Model Card के metadata के साथ एक [`ModelCardData`] instance लौटाता है। इसे dictionary के रूप में पाने के लिए इस instance पर `.to_dict()` कॉल करें।
  - `card.text`: card का text लौटाता है, *metadata header को छोड़कर*।
  - `card.content`: card का text content लौटाता है, *metadata header सहित*।

## Model Cards बनाएँ

### Text से

Text से Model Card बनाने के लिए, init के समय card का text content बस `ModelCard` को पास कर दें।

```python
content = """
---
language: en
license: mit
---

# My Model Card
"""

card = ModelCard(content)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True
```

ऐसा करने का एक और तरीका f-strings है। नीचे दिए गए उदाहरण में हम:

- अपने तय किए हुए metadata को YAML में बदलने के लिए [`ModelCardData.to_yaml`] का उपयोग करते हैं, ताकि model card में YAML block डाला जा सके।
- दिखाते हैं कि Python f-strings के ज़रिए template variable का उपयोग कैसे किया जा सकता है।

```python
card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
content = f"""
---
{ card_data.to_yaml() }
---

# My Model Card

This model created by [@{example_template_var}](https://github.com/{example_template_var})
"""

card = ModelCard(content)
print(card)
```

ऊपर के उदाहरण से हमें ऐसा card मिलेगा:

```
---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)
```

### Jinja Template से

अगर आपके पास `Jinja2` इंस्टॉल है, तो आप jinja template file से Model Cards बना सकते हैं। आइए एक बुनियादी उदाहरण देखें:

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Define your jinja template
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Write the template to a file
Path('custom_template.md').write_text(template_text)

# Define card metadata
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Create card from template, passing it any jinja template variables you want.
# In our case, we'll pass author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

नतीजे में बने card का markdown ऐसा दिखता है:

```
---
language: en
license: mit
library_name: keras
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

अगर आप card.data में कुछ भी अपडेट करते हैं, तो वह card में भी दिखाई देगा।

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

अब, जैसा कि आप देख सकते हैं, metadata header अपडेट हो गया है:

```
---
language: fr
license: apache-2.0
library_name: timm
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

card data अपडेट करते समय, आप [`ModelCard.validate`] कॉल करके जाँच सकते हैं कि card अब भी Hub के अनुसार मान्य है। इससे यह सुनिश्चित होता है कि card Hugging Face Hub पर तय किए गए सभी validation नियमों को पास करता है।

### Default Template से

अपना खुद का template इस्तेमाल करने की जगह आप [default template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md) भी इस्तेमाल कर सकते हैं। यह एक पूरी सुविधाओं वाला model card है जिसमें भरने के लिए ढेर सारे sections हैं। अंदर से यह template file भरने के लिए [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) का उपयोग करता है।

> [!TIP]
> ध्यान दें कि `from_template` इस्तेमाल करने के लिए आपके पास Jinja2 इंस्टॉल होना ज़रूरी है। आप इसे `pip install Jinja2` से इंस्टॉल कर सकते हैं।

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
card.save('my_model_card_2.md')
print(card)
```

## Model Cards शेयर करें

अगर आप Hugging Face Hub पर authenticated हैं (`hf auth login` या [`login`] के ज़रिए), तो आप बस [`ModelCard.push_to_hub`] कॉल करके cards को Hub पर push कर सकते हैं। आइए देखें यह कैसे किया जाता है...

सबसे पहले, हम authenticated user के namespace में 'hf-hub-modelcards-pr-test' नाम का एक नया repo बनाएँगे:

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

फिर, हम default template से एक card बनाएँगे (बिल्कुल वैसा ही जैसा ऊपर के section में तय किया गया था):

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
```

आखिर में, हम इसे Hub पर push कर देंगे

```python
card.push_to_hub(repo_id)
```

नतीजे में बना card आप [यहाँ](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md) देख सकते हैं।

अगर आप card को pull request के रूप में push करना चाहते हैं, तो `push_to_hub` कॉल करते समय बस `create_pr=True` दें:

```python
card.push_to_hub(repo_id, create_pr=True)
```

इस command से बना एक PR [यहाँ](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3) देखा जा सकता है।

## Metadata अपडेट करें

इस section में हम देखेंगे कि repo cards में metadata क्या होता है और उसे कैसे अपडेट किया जाता है।

`metadata` एक hash map (यानी key-value) होता है जो किसी model, dataset या Space के बारे में कुछ उच्च-स्तरीय जानकारी देता है। इस जानकारी में model का `pipeline type`, `model_id` या `model_description` जैसी बातें शामिल हो सकती हैं। ज़्यादा जानकारी के लिए आप ये guides देख सकते हैं: [Model Card](https://huggingface.co/docs/hub/model-cards#model-card-metadata), [Dataset Card](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) और [Spaces Settings](https://huggingface.co/docs/hub/spaces-settings#spaces-settings)।
अब देखते हैं कि इस metadata को अपडेट कैसे किया जाता है, कुछ उदाहरणों के साथ।


आइए पहले उदाहरण से शुरू करते हैं:

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

code की इन दो पंक्तियों से आप metadata अपडेट करके एक नया `pipeline_tag` सेट कर देंगे।

डिफ़ॉल्ट रूप से, आप card पर पहले से मौजूद किसी key को अपडेट नहीं कर सकते। ऐसा करना हो तो आपको
`overwrite=True` साफ़ तौर पर पास करना होगा:


```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

अक्सर ऐसा होता है कि आप किसी ऐसे repository में बदलाव सुझाना चाहते हैं
जिस पर आपके पास write permission नहीं है। आप उस repo पर एक PR बनाकर ऐसा कर सकते हैं, जिससे मालिक आपके सुझावों को
review करके merge कर सकेंगे।

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## Evaluation Results शामिल करें

metadata के `model-index` में evaluation results शामिल करने के लिए, आप अपने evaluation results के साथ एक [`EvalResult`] या `EvalResult` की list पास कर सकते हैं। अंदर से यह तब `model-index` बनाता है जब आप `card.data.to_dict()` कॉल करते हैं। यह कैसे काम करता है, इसकी ज़्यादा जानकारी के लिए आप [Hub docs का यह section](https://huggingface.co/docs/hub/models-cards#evaluation-results) देख सकते हैं।

> [!TIP]
> ध्यान दें कि इस function का उपयोग करने के लिए आपको [`ModelCardData`] में `model_name` attribute शामिल करना होगा।

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = EvalResult(
        task_type='image-classification',
        dataset_type='beans',
        dataset_name='Beans',
        metric_type='accuracy',
        metric_value=0.7
    )
)

card = ModelCard.from_template(card_data)
print(card.data)
```

नतीजे में बना `card.data` ऐसा दिखना चाहिए:

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
```

अगर आपके पास शेयर करने के लिए एक से ज़्यादा evaluation result हैं, तो बस `EvalResult` की list पास करें:

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = [
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.7
        ),
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='f1',
            metric_value=0.65
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data
```

इससे आपको यह `card.data` मिलेगा:

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65
```
