## Use Cases

Use Case: Domain Adaptation
Masked language modeling is used to solve domain-specific problems.
If you have to work on a domain-specific task, let’s say, information retrieval from medical papers. You can train a masked language model from medical papers, and then fine-tune on a downstream task, such as Question Natural Language Inference, or Question Answering, to build a medical information extraction system. Pre-training on domain-specific data yields better results. (cite: https://arxiv.org/abs/2007.15779) You can also use a [domain-specific BERT model](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) from the Hub and fine-tune on task-specific dataset as well.

## Inference with Fill-Mask Pipeline

You can use “fill-mask” pipeline to infer with masked language models. If model name is not provided, the pipeline will be initialized with “distilroberta-base”. You can provide masked text and it will return list of possible mask values, ranked according to the score.

```python
from transformers import pipeline

classifier = pipeline("fill-mask")
classifier("Paris is the <mask> of France.")```
```



## Useful Resources
[Course Chapter on Fine-tuning a Masked Language Model](https://huggingface.co/course/chapter7/3?fw=pt)
### Notebooks
[Pre-training an MLM for JAX/Flax](https://github.com/huggingface/notebooks/blob/master/examples/masked_language_modeling_flax.ipynb)
[Masked language modeling in TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling-tf.ipynb)
[Masked language modeling in PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb)
### Scripts
[PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)
[Flax](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling)
[TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/language-modeling)