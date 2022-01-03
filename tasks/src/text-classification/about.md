## Use Cases 

### Sentiment Analysis on Customer Reviews
You can track your customers’ sentiments from the product reviews using sentiment analysis models. This can help understand the churn and retention by grouping the reviews according to sentiments, analyzing the text and taking strategic decisions based on these insights.

## Task Variants 

### Natural Language Infenrence (NLI)

In NLI, the model determines the relationship between two given texts. Concretely, the model takes a premise and a hypothesis and returns a class that can either be:
* **entailment**, which means the hypothesis is true.
* **contraction**, which means the hypothesis is false.
* **neutral**, which means there's no relation between the hypothesis and the premise.

The benchmark dataset for this task is GLUE (General Language Understanding Evaluation). NLI models have different variants, such as Multi-Genre NLI, Question NLI and Winograd NLI. 

### Multi-Genre NLI (MNLI)

MNLI is used for general NLI. Here are som examples:

```
Example 1: 
    Premise: A man inspects the uniform of a figure in some East Asian country.
    Hypothesis: The man is sleeping.
    Label: Contradiction

Example 2: 
    Premise: Soccer game with multiple males playing.
    Hypothesis: Some men are playing a sport.
    Label: Entailment
``` 

### Inference
You can use the 🤗 Transformers library `text-classification` pipeline to infer with NLI models.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "roberta-large-mnli")
classifier("A soccer game with multiple males playing. Some men are playing a sport.")
## [{'label': 'ENTAILMENT', 'score': 0.98}]
```

### Question Natural Language Inference (QNLI)
QNLI is the task of determining if the answer of a given question can be found in a given document. If the answer can be found, the label is “entailment” and if the answer cannot be found, it’s “not entailment".

```
Question: What percentage of marine life died during the extinction?
Sentence: It is also known as the “Great Dying” because it is considered the largest mass extinction in the Earth’s history.
Label: Not entailment

Question: Who was the London Weekend Television’s Managing Director? 
Sentence: The managing director of London Weekend Television (LWT), Greg Dyke, met with the representatives of the "big five" football clubs in England in 1990.
Label: Entailment
```

#### Inference
You can use the 🤗 Transformers library `text-classification` pipeline to infer with QNLI models. Just as before, the model returns the label and the confidence.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "cross-encoder/qnli-electra-base")
classifier("Where is the capital of France?, Paris is the capital of France.")
# [{'label': 'entailment', 'score': 0.997}] 
```

### Sentiment Analysis
In sentiment analysis, the classes can be polarities like positive, negative, neutral, or sentiments such as happiness or anger. 

You can use the 🤗 Transformers library with the `sentiment-analysis` pipeline to infer with sentiment analysis models. The model returns the label with the score.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I loved Star Wars so much!") 
##  [{'label': 'POSITIVE', 'score': 0.99}
```

### Quora Question Pairs

Quora Question Pairs is a task that assesses if two provided questions are paraphrases of each other. The model takes two questions and returns a binary value, with 0 being mapped to “not paraphrase” and 1 being “paraphrase". The benchmark dataset is Q[uora Question Pairs](https://huggingface.co/datasets/glue/viewer/qqp/test) inside the [GLUE benchmark](https://huggingface.co/datasets/glue). The dataset consists of question pairs and their labels.

```
Question1: “How can I increase the speed of my internet connection while using a VPN?” 
Question2: How can Internet speed be increased by hacking through DNS?
Label: Not paraphrase

Question1: “What can make Physics easy to learn?”
Question2: “How can you make physics easy to learn?”
Label: Paraphrase
```

#### Inference
You can use the 🤗 Transformers library `text-classification` pipeline to infer with QQPI models.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "textattack/bert-base-uncased-QQP")
classifier("Which city is the capital of France?, Where is the capital of France?")
## [{'label': 'paraphrase', 'score': 0.998}]
```

### Grammatical Correctness
Linguistic Acceptability is the task of assessing the grammatical acceptability of a sentence. The classes in this task are “acceptible” and “unacceptable”. The benchmark dataset used for this task is [Corpus of Linguistic Acceptability (CoLA)](https://huggingface.co/datasets/glue/viewer/cola/test). The dataset consists of texts and their labels.

```
Example: Books were sent to each other by the students.
Label: Unacceptable

Example: She voted for herself.
Label: Acceptable.
```

#### Inference

```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "textattack/distilbert-base-uncased-CoLA")
classifier("I will walk to home when I went through the bus.")
##  [{'label': 'unacceptable', 'score': 0.95}]
```


## Useful Resources
Would you like to learn more about the topic? Awesome! Here you can find some curated resources that can be helpful to you!
- [Course Chapter on Fine-tuning a Text Classification Model](https://huggingface.co/course/chapter3/1?fw=pt)

### Notebooks
- [PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb)
- [TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/text_classification-tf.ipynb)
- [Flax](https://github.com/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb)

### Scripts for training
- [PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification)
- [TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/text-classification)
- [Flax](https://github.com/huggingface/transformers/tree/master/examples/flax/text-classification)