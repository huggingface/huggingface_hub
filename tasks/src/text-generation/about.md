## Use Cases
We’ll cover both models for the use cases.
If the generative model’s training data is different than your use case, you can train a causal language model from scratch. This is taught in the chapter 7 of the Hugging Face Course. 
Training a causal language model on code from scratch will help the programmers in their repetitive coding tasks. 
Another interesting use cases for text generation models are generating legal documents and generating stories.

## Inference
Text generation models can be inferred with “text-generation” pipeline.
“text-generation” pipeline takes an incomplete text and returns multiple outputs of which the text can be completed with. If the pipeline is called with no model name, it will be initialized with GPT-2 by default.

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
```

Text-to-Text generation models have a separate pipeline called "text2text-generation". If no model checkpoint is given, the pipeline is initialized using “t5-base”. This pipeline takes an input containing the sentence including the task and returns the output of the accomplished task.

```python
from transformers import pipeline

text2text_generator = pipeline("text2text-generation")
text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
[{'generated_text': 'the answer to life, the universe and everything'}]

text2text_generator("translate from English to French: I'm very happy")
[{'generated_text': 'Je suis très heureux'}]
```
T0 model is more robust and flexible on task prompts. 
```python
text2text_generator = pipeline("text2text-generation", model = "bigscience/T0")

text2text_generator(“Is the word 'table' used in the same meaning in the two previous sentences? Sentence A: you can leave the books on the table over there. Sentence B: the tables in this book are very hard to read.” )
## [{"generated_text": "No"}]

text2text_generator(“A is the son's of B's brother. What is the family relationship between A and B?”)
## [{"generated_text": "brother"}]

text2text_generator(“Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy”)
## [{"generated_text": "positive"}]

text2text_generator(“Reorder the words in this sentence: justin and name bieber years is my am I 27 old.”)
##  [{"generated_text": "justin bieber is my name and i am 27 years old"}]
```


## Useful Resources
- [Course Chapter on Training a causal language model from scratch](https://huggingface.co/course/chapter7/6?fw=pt)
### Notebooks
- [Training a CLM in Flax](https://github.com/huggingface/notebooks/blob/master/examples/causal_language_modeling_flax.ipynb)
- [Training a CLM in TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch-tf.ipynb)
- [Training a CLM in PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb)
### Scripts
- [Training a CLM in PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)
- [Training a CLM in TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/language-modeling)
- [Text Generation in PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation)