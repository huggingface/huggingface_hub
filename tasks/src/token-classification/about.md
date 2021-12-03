## Use Case: Information Extraction from Invoices
You can extract important entities from invoices automatically using Named Entity Recognition models. Invoices can be read with Optical Character Recognition models, and the output of this can be used to do inference with Named Entity Recognition models. This way, important information, such as date, company name and other named entities can be extracted.

## Task Variants

### Named Entity Recognition
Named entity recognition is the task of recognizing named entities in a text. These entities can be person, location or organization names. The task is formulated as labelling each token with one class for each named entity and a class called “O” for tokens that contain no entity. The input of this task is a text and output is the annotated text with named entities.

#### Inference 
You can use “ner” pipeline to infer with Named Entity Recognition models. If you don’t provide any model name, the pipeline will be initialized with the BERT fine-tuned on ConLL03, “dbmdz/bert-large-cased-finetuned-conll03-english”
```python
from transformers import pipeline

classifier = pipeline("ner")
classifier("Hello I'm Omar and I live in Zürich.")
```

### Part-of-Speech Tagging
Part-of-Speech tagging is to recognize parts of speech in a given text. The task is formulated as labelling the words for a particular part of a speech, such as noun, pronoun, adjective, verb and so on. 

#### Inference
As of now, there’s no default model or specific pipeline for POS tagging so you can use “token-classification” pipeline with a POS tagging model of your choice. The model will return a json with part-of-speech tags for each token.
```python
from transformers import pipeline

classifier = pipeline("token-classification", model = "vblagoje/bert-english-uncased-finetuned-pos")
classifier("Hello I'm Omar and I live in Zürich.")
```



## Useful Resources
- [Course Chapter on Token Classification](https://huggingface.co/course/chapter7/2?fw=pt)

### Token Classification Notebooks
- [PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)
- [TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/token_classification-tf.ipynb)

### Token Classification Scripts
- [PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification)
- [TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow)
- [Flax](https://github.com/huggingface/transformers/tree/master/examples/flax/token-classification)