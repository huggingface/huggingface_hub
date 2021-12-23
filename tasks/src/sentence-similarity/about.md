## Use Cases 🔍

You can extract information from documents using sentence similarity models. The first step is to rank documents using Passage Ranking models. You can then get the top-ranked document and search in it with sentence similarity models by picking the sentence that has the highest similarity to the input query.

## Sentence Transformers
The [Sentence Transformers](https://www.sbert.net/) library is very powerful to compute embeddings of sentences, paragraphs and whole documents. An embedding is just a vector representation of a text, which makes embeddings very useful to find how similar two texts are. 

You can find and use [hundreds of Sentence Transformers](https://huggingface.co/models?library=sentence-transformers&sort=downloads) models from the Hub
by directly using the library, playing with the widgets in the browser, or using the Inference API.

## Task Variants

### Passage Ranking
Passage ranking is the task of ranking documents based on their relevance to a given query. The task is evaluated on Mean Reciprocal Rank. These models take one query and multiple documents and return ranked documents according to the relevancy to the query. 📄

You can infer with passage ranking models using the [Inference API](https://huggingface.co/inference-api). The inputs to the passage ranking model is a source sentence for which we are looking and the documents we want to search in. The model will return scores according to relevancy of those documents to the source sentence. 

```python
import json
import requests

# msmarco models are used for passage ranking
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b" 
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query(
    {
        "inputs": {
            "source_sentence": "That is a happy person",
            "sentences": [
                "That is a happy dog",
                "That is a very happy person",
                "Today is a sunny day"
            ]
        }
    }
## [0.853, 0.981, 0.655]
```

### Semantic Textual Similarity
Semantic textual similarity is the task of assessing how similar two texts are in terms of meaning. The task is evaluated on Pearson’s Rank Correlation. These models take one source sentence and a list of sentences that we will look for similarity in and return a list of similarity scores. The benchmark dataset is the [Semantic Textual Similarity Benchmark](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark).

```python
import json
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2" # sentence similarity model
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query(
    {
        "inputs": {
            "source_sentence": "I'm very happy",
            "sentences":["I'm filled with happiness", "I'm happy"]
        }
    })

## [0.605, 0.894]
```

You can also infer with the models in the Hub using Sentence Transformer models.

```python
pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer, util
sentences = ["I'm happy", "I'm full of happiness"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

util.pytorch_cos_sim(embedding_1, embedding_2)
## tensor([[0.6003]])
```


## Useful Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Sentence Transformers in the Hub](https://huggingface.co/blog/sentence-transformers-in-the-hub)
