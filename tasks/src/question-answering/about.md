## Use Cases

You can use question answering models with knowledge base as contexts and user inputs as questions to automate frequently asked questions. You need documents containing your company’s informations and a question answering model either from the Hub or trained by you. Questions coming from your customers can be inferred from those documents. If you’d like to save inference time, you can first use passage ranking models to see which document might contain the answer to the question and iterate over that document with the question answering model instead.

## Task Variants
There are two question answering types, one is open-domain question answering, and the other one is closed domain question answering. Open-domain question answering models are not restricted to a specific domain, meanwhile closed-domain questions answering models are restricted to a specific domain (e.g. legal, medical documents). According to taking context information, there are two types of question answering problems. Open book question answering models take a context with a question where answer is extracted by the context. Closed book question answering models takes no context and are expected to know answer to question, and generate the answer rather than extracting. There are various question answering variants based on the input and output they return, being, extractive question answering (that is solved with BERT-like models), generative question answering (that is solved with models like BART and T5) and Table Question Answering (solved with TAPAS model).
The below schema illustrates extractive, open book question answering. The model takes a context and the question and extract the answer from the given context.

## Inference

You can infer with Question Answering models with the “question-answering” pipeline. If no model checkpoint is given, the pipeline will be initialized with “distilbert-base-cased-distilled-squad”. Question answering pipeline takes a context to be searched in and a question to be searched for an answer and returns an answer.
```python
from transformers import pipeline

qa_model = pipeline("question-answering")
question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
qa_model(question = question, context = context)
## {'answer': 'İstanbul', 'end': 39, 'score': 0.9538118243217468, 'start': 31}
```


## Useful Resources
- [Course Chapter on Question Answering](https://huggingface.co/course/chapter7/7?fw=pt)
### Notebooks on Question Answering
- [PyTorch](https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb)
- [TensorFlow](https://github.com/huggingface/notebooks/blob/master/examples/token_classification-tf.ipynb)
### Scripts on Question Answering
- [PyTorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering)
- [TensorFlow](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/question-answering)
- [Flax](https://github.com/huggingface/transformers/tree/master/examples/flax/question-answering)
- [Blog Post: ELI5 A Model for Open Domain Long Form Question Answering](https://yjernite.github.io/lfqa.html)