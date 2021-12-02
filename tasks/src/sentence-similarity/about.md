## Use Cases
You can extract information from documents using sentence similarity models. First you can rank the documents themselves with Passage Ranking models, get the best performing document and search in it with sentence similarity models, pick the sentence with best cosine similarity. 

## Task Variants

### Passage Ranking
Passage ranking is the task of ranking documents based on their relevance to a given query. The task is evaluated on Mean Reciprocal Rank. Passage Ranking models take one query and multiple documents and return ranked documents according to the relevancy to the query.

You can infer with passage ranking models using the Inference API. The inputs to the passage ranking model is a source sentence for which we are looking for the relevant documents and the documents we want to search in. The model will return scores according to relevancy of those documents to the source sentence. 

```python
import json
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b" # msmarco models are used for passage ranking
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
## [0.853405773639679, 0.9814600944519043, 0.6550564765930176]
```

### Semantic Textual Similarity
Semantic textual similarity is the task of assessing how similar two texts are. The task is evaluated on Pearson’s Rank Correlation. Semantic textual similarity models take one source sentence and a list of sentences that we will look for similarity in and return a list of similarity scores. The benchmark dataset is The Semantic Textual Similarity Benchmark.

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

## [0.6058085560798645, 0.8944037556648254]
```
You can also infer with the models in the Hub through sentence-transformers.

```python
pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer, util
sentences = ["I'm happy", "I'm full of happiness"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

cosine_score = util.pytorch_cos_sim(embedding_1, embedding_2)
print(cosine_score)
## tensor([[0.6003]])
```


## Useful Resources
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Sentence Transformers in the Hub](https://huggingface.co/blog/sentence-transformers-in-the-hub)
