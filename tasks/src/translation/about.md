You can find over a thousand translation models on the Hub, but sometimes you might not find a model for the pair of languages you're interested in. When this happens, you can use a pretrained multilingual translation model, like [mBART](https://huggingface.co/facebook/mbart-large-cc25), and further train it with your own data in a process called fine-tuning.

## Use Cases 
### Multilingual conversational agents
Translation models can be used to build conversational agents across different languages. This can be done in two ways.

* **Translate dataset to a new language.** You can translate a dataset of intents (inputs) and responses to the target language. You can then train a new intent classification model with this new dataset. This allows you to proofread responses in the target language and have better control of chatbot's outputs.
- **Translate input and output of the agent.** You can use a translation model to translate the user inputs in a way that the chatbot can process it. You can then translate the output of the chatbot to the user language. This approach might be less reliable since chatbot will output responses that are not defined before.

## Inference
You can use the 🤗 Transformers library with the `translation_xx_to_yy` pattern where xx is the source language code and yy is the target language code. The default model for the pipeline is [t5-base](https://huggingface.co/t5-base), which under the hood adds a task prefix indicating the task itself, e.g. “translate: English to French”.

```python
from transformers import pipeline
en_fr_translator = pipeline("translation_en_to_fr")
en_fr_translator("How old are you?")
## [{'translation_text': ' quel âge êtes-vous?'}]
```

If you’d like to use a specific model checkpoint that is from one specific language to another, you can also directly use the “translation” pipeline. 

```python
from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("How are you?")
# [{'translation_text': 'Comment allez-vous ?'}]
```


## Useful Resources
Would you like to learn more about translation? Great! Here you can find some curated resources that can be helpful to you!

- [Course Chapter on Translation](https://huggingface.co/course/chapter7/4?fw=pt)

### Notebooks
- [PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb)
- [TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/translation-tf.ipynb)

### Scripts for training
- [PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation)
- [TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/translation)