## Use Cases
A text generation model, also known as causal language model, can be trained on code from scratch to help the programmers in their repetitive coding tasks. Another interesting use cases for text generation models are generating stories.

If the generative model’s training data is different than your use case, you can train a causal language model from scratch. Learn how to do it in the free transformers [course](https://huggingface.co/course/chapter7/6?fw=pt)!

## Task Variants

A popular variant of text generation models predicts the next word given a bunch of words. For example:
* Given an incomplete sentence, complete it.
* Continue a story given the first sentences.
* Provided a code description, generate the code.

The most popular models for this are GPT-based models (such as [GPT-2](https://huggingface.co/gpt2)). These models are trained on data that has no labels, so you just need plain text to train your own model. You can train GPT models to generate a wide range of documents, from code to story. 

Another variant of generative models is called “text-to-text” generative model. This is trained to learn mapping between a pair of text (e.g. translation from one language to another). The most popular variants of this model are [T5](https://huggingface.co/docs/transformers/model_doc/t5), [T0](https://huggingface.co/bigscience/T0pp) and [BART](https://huggingface.co/docs/transformers/model_doc/bart). These models are trained with multi-tasking capabilities, they can accomplish a wide range of tasks, including summarization, translation, and text classification.

## Inference
You can use the 🤗 Transformers library `text-generation` pipeline to do inference with text generation models. It takes an incomplete text and returns multiple outputs of which the text can be completed with. 

```python
from transformers import pipeline
generator = pipeline('text-generation', model = 'gpt2')
generator("Hello, I'm a language model", max_length = 30, num_return_sequences=3)
## [{'generated_text': "Hello, I'm a language modeler. So while writing this, when I went out to meet my wife or come home she told me that my"},
##  {'generated_text': "Hello, I'm a language modeler. I write and maintain software in Python. I love to code, and that includes coding things that require writing"}, ...
```

Text-to-Text generation models have a separate pipeline called `text2text-generation`. This pipeline takes an input containing the sentence including the task and returns the output of the accomplished task.

```python
from transformers import pipeline

text2text_generator = pipeline("text2text-generation")
text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
[{'generated_text': 'the answer to life, the universe and everything'}]

text2text_generator("translate from English to French: I'm very happy")
[{'generated_text': 'Je suis très heureux'}]
```

The [T0 model](https://huggingface.co/bigscience/T0) is even more robust and flexible on task prompts. 

```python
text2text_generator = pipeline("text2text-generation", model = "bigscience/T0")

text2text_generator(“Is the word 'table' used in the same meaning in the two previous sentences? Sentence A: you can leave the books on the table over there. Sentence B: the tables in this book are very hard to read.” )
## [{"generated_text": "No"}]

text2text_generator(“A is the son's of B's brother. What is the family relationship between A and B?”)
## [{"generated_text": "brother"}]

text2text_generator(“Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy”)
## [{"generated_text": "positive"}]

text2text_generator(“Reorder the words in this sentence: justin and name bieber years is my am I 27 old.”)
##  [{"generated_text": "Justin Bieber is my name and I am 27 years old"}]
```


## Useful Resources
Would you like to learn more about the topic? Awesome! Here you can find some curated resources that can be helpful to you!
- [Course Chapter on Training a causal language model from scratch](https://huggingface.co/course/chapter7/6?fw=pt)
- [TO Discussion with Victor Sanh](https://www.youtube.com/watch?v=Oy49SCW_Xpw&ab_channel=HuggingFace)
- [Hugging Face Course Workshops: Pretraining Language Models & CodeParrot](https://www.youtube.com/watch?v=ExUR7w6xe94&ab_channel=HuggingFace)
- [Training CodeParrot 🦜 from Scratch](https://huggingface.co/blog/codeparrot)
- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)

### Notebooks
- [Training a CLM in Flax](https://github.com/huggingface/notebooks/blob/master/examples/causal_language_modeling_flax.ipynb)
- [Training a CLM in TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch-tf.ipynb)
- [Training a CLM in PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb)

### Scripts for training
- [Training a CLM in PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)
- [Training a CLM in TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/language-modeling)
- [Text Generation in PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation)