## Use Case: Information Extraction from Invoices
Translation models can be used to build conversational agents across different languages. This can be done in two ways.
One option is to translate the training data of intent classification algorithm and the responses defined for each intent from source language of existing data to target language, and train a new intent classification and dialogue management models. You can proofread the responses in the target language to provide better control over your chatbot’s outputs.
Another way is to put one translation model from the target language to the language the chatbot is trained on, this will translate the user inputs. The output of this will be input to intent classification algorithm in the source language. After predicting the user intent, we take the classified intent and the response to that intent in the source language. We take the output and translate it to user’s language. This approach might be less reliable since chatbot will output responses that are not defined before.

## Inference
Translation models are loaded with “translation_xx_to_yy” pattern where xx is the source language code and yy is the target language code. Default model for the pipeline is “t5-base”.  If you’re directly inferring with the T5 model, the model will expect a task prefix indicating the task itself, e.g. “translate: English to French”.

```python
from transformers import pipeline
en_fr_translator = pipeline("translation_en_to_fr")
en_fr_translator("How old are you?")

If you’d like to use a specific model checkpoint that is from one specific language to another, you can also directly use the “translation” pipeline. 

from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("How are you?")

```



## Useful Resources

- [Course Chapter on Translation](https://huggingface.co/course/chapter7/4?fw=pt)

### Translation Notebooks
- [PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb)
- [TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/translation-tf.ipynb)

### Translation Scripts
- [PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation)
- [TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/translation)