Text classification models have many variants, including, sentiment analysis, natural language inference, linguistic acceptibility and more. We will go through all of them. The largest variant is Natural Language Inference (NLI). NLI is to infer between a premise and a hypothesis, and NLI models take a premise and a hypothesis and return a label with possible values of contraction, neutral, and entailment. If the hypothesis is true, NLI model returns “entailment” (positive/true), if the hypothesis is false, it returns “contradiction” (negative/false) or there’s no relation, it returns “neutral”. The benchmark dataset for this task is GLUE (General Language Understanding Evaluation). NLI models have different variants, Multi-Genre NLI, Question NLI and Winograd NLI. 

## Use Cases 

### Sentiment Analysis on Customer Reviews
You can track your customers’ sentiments over time from the product reviews using sentiment analysis models and understand the churn and retention by grouping your reviews according to sentiments, analyze text and build a strategy accordingly.

## Task Variants 
There are three variants of Natural Language Inference, Multi Genre Natural Language Inference (MNLI),  Winograd Natural Language Inference and Question Natural Language Inference (QNLI).

### MultiNLI
MNLI is modeled after Stanford Natural Language Inference (SNLI), and both MNLI and SNLI are for general natural language inference. 
```
Premise: A man inspects the uniform of a figure in some East Asian country.
Hypothesis: The man is sleeping.
Label: Contradiction
Example: 
Premise: Soccer game with multiple males playing.
Hypothesis: Some men are playing a sport.
Label: Entailment.
``` 

#### Inference
You can use “text-classification” pipeline to infer with NLI models.
```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "roberta-large-mnli")
classifier("A soccer game with multiple males playing. Some men are playing a sport.")
## [{'label': 'ENTAILMENT', 'score': 0.98}]
```

### Question Natural Language Inference
Question Natural Language Inference (QNLI) is the task of determining if the answer of a given question can be found in a given document. If the answer can be found, the label is “entailment” and if the answer cannot be found, it’s “not entailment.
```
Question: What percentage of marine life died during the extinction?
Sentence: It is also known as the “Great Dying” because it is considered the largest mass extinction in the Earth’s history.
Label: Not entailment
Question: Who was the London Weekend Television’s Managing Director? 
Sentence: The managing director of London Weekend Television (LWT), Greg Dyke, met with the representatives of the "big five" football clubs in England in 1990.
Label: Entailment
```

#### Inference
As of now, you can use “text-classification” pipeline to infer with QNLI models. The model returns one of the labels and the confidence.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "cross-encoder/qnli-electra-base")
classifier("Where is the capital of France?, Paris is the capital of France.")
# [{'label': 'entailment', 'score': 0.997}] 
```

### Winograd Natural Language Inference
Winograd Natural Language Inference (WNLI) is based on Winograd Schema Challenge, which is the task of reading comprehension where the model reads a sentence with pronoun and is expected to assign a pronoun referring to the given pronoun. 
```
Sentence: Susan knew that Ann's son had been in a car accident, so she told her about it. 
Sentence: Susan told her about it.
Label: Entailment

Example: 
Sentence: Carol believed that Rebecca suspected that she had stolen the watch.
Sentence: Carol believed that Rebecca suspected that Rebecca had stolen the watch.
Label: Not entailment
```

#### Inference
WNLI models can be inferred with “text-classification” pipeline. Inference returns a label and the score associated with that label.
```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "textattack/bert-base-uncased-WNLI")
classifier("I put the cake away in the refrigerator. It has a lot of butter in it., The refrigerator has a lot of butter in it.")
## [{'label': 'entailment', 'score': 0.529}]
```

### Sentiment Analysis
Sentiment Analysis is determining the sentiment of a given text. The classes can be polarities like positive, negative, neutral, or sentiments like happiness, anger. The task is evaluated on Stanford Sentiment Treebank.
There are variants of the sentiment analysis task. Aspect based sentiment analysis is the task of evaluating the aspect of an attribute of an entity, this can be a product review, where different aspects can be included in the same text. 
```
I would recommend the product, it is good quality but the price is high. 
```
Above example contains a review with quality and price aspects and contains multiple sentiments.
Another variant of sentiment analysis is called Multimodal Sentiment Analysis is and it is the task of classifying a sentiment based on visual data or speech.

#### Inference
You can directly use “sentiment-analysis” pipeline to infer with sentiment analysis models. The model returns the label with the score.
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I loved Star Wars so much!") 
##  [{'label': 'POSITIVE', 'score': 0.99}
```

### Quora Question Pairs
Quora Question Pairs is a question answering related task that assesses if given two questions are paraphrases of each other. The model takes two questions and returns a binary value, with 0 being mapped to “not paraphrase” and 1 being “paraphrase. The benchmark dataset is Quora Question Pairs inside the GLUE benchmark. The dataset consists of question pairs and their labels.
```
Question1: “How can I increase the speed of my internet connection while using a VPN?” Question2: How can Internet speed be increased by hacking through DNS?
Label: Not paraphrase
Question1: “What can make Physics easy to learn?”
Question2: “How can you make physics easy to learn?”
Label: Paraphrase
```

#### Inference
You can use “text-classification” pipeline to infer with QQP models.
```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "textattack/bert-base-uncased-QQP")
classifier("Which city is the capital of France?, Where is the capital of France?")
## [{'label': 'entailment', 'score': 0.998}]
```

### Linguistic Acceptibility
Linguistic Acceptibility is the task of assessing the grammatical acceptibility of a sentence. The classes in this task are “acceptible” and “unacceptable”. The benchmark dataset used for this task is Corpus of Linguistic Acceptibility (CoLA). The dataset consists of texts and their labels.
```
Example: Books were sent to each other by the students.
Label: Unacceptable
Example: She voted for herself.
Label: Acceptable.
```

#### Inference
You can use “text-classification” pipeline to infer with linguistic acceptibility models.
```python
from transformers import pipeline

classifier = pipeline("text-classification", model = "textattack/distilbert-base-uncased-CoLA")
classifier("I will walk to home when I went through the bus.")
##  [{'label': 'unacceptable', 'score': 0.95}]
```


## Useful Resources

### Notebooks on Text Classification
- [PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb)
- [TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/text_classification-tf.ipynb)
- [Flax](https://github.com/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb)

### Scripts for Text Classification
- [PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification)
- [TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/text-classification)
- [Flax](https://github.com/huggingface/transformers/tree/master/examples/flax/text-classification)