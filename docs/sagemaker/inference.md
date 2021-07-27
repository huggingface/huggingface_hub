---
title: Deploy models to Amazon SageMaker
---

<h1>Deploy models to Amazon SageMaker</h1>

Here's how easy it is to use the new [SageMaker Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) to deploy 🤗 Transformers models in SageMaker:

```python
from sagemaker.huggingface import HuggingFaceModel

# create Hugging Face Model Class and deploy it as SageMaker Endpoint
huggingface_model = HuggingFaceModel(...).deploy()
```

## Overview

- [SageMaker Hugging Face Inference Toolkit](#sagemaker-hugging-face-inference-toolkit)
- [Inference Toolkit - API Description](#inference-toolkit---api-description)
  - [Setup & Installation](#setup--installation)
- [Deploy for inference a 🤗 Transformers model trained in SageMaker](#deploy-for-inference-a-🤗-transformers-model-trained-in-sagemaker)
  - [Deploy the model directly after training](#deploy-the-model-directly-after-training)
  - [Deploy the model using `model_data`](#deploy-the-model-using-model_data)
- [Deploy for inference one of the 10,000+ 🤗 Transformers models available in the 🤗 Hub](#deploy-for-inference-one-of-the-10000-%F0%9F%A4%97-transformers-models-available-in-the-%F0%9F%A4%97-hub)
- [Advanced Features](#advanced-features)
  - [Environment variables](#environment-variables)
  - [Creating a Model artifact model.tar.gz for deployment](#creating-a-model-artifact-modeltargz-for-deployment)
  - [User defined code/modules](#user-defined-codemodules)
  - [Batch transform](#batch-transform)
- [Additional Resources](#additional-resources)


## [SageMaker Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit)

In addition to the Hugging Face Inference Deep Learning Containers, we created a new [Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) for SageMaker. This new Inference Toolkit leverages the `pipelines` from the `transformers` library to allow zero-code deployments of models, without requiring any code for pre- or post-processing. In the "Getting Started" section below, you will find two examples of how to deploy your models to Amazon SageMaker.

In addition to zero-code deployment, the Inference Toolkit supports "bring your own code" methods, where you can override the default methods. You can learn more about "bring your own code" in the documentation[ here](https://github.com/aws/sagemaker-huggingface-inference-toolkit#-user-defined-codemodules), or you can check out the sample notebook "deploy custom inference code to Amazon SageMaker".

## Inference Toolkit - API Description

Using the `transformers pipelines`, we designed an API, which makes it easy for you to benefit from all `pipelines` features. The API has a similar interface than the [🤗 Accelerated Inference API](https://api-inference.huggingface.co/docs/python/html/detailed_parameters.html) hosted service: your inputs need to be defined in the `inputs` key, and additional supported `pipelines` parameters can be added in the `parameters` key. Below, you can find request examples.

**`text-classification`**
**`sentiment-analysis`**
**`token-classification`**
**`feature-extraction`**
**`fill-mask`**
**`summarization`**
**`translation_xx_to_yy`**
**`text2text-generation`**
**`text-generation`**

```json
{
  "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
}
```

**`question-answering`**

```json
{
  "inputs": {
    "question": "What is used for inference?",
    "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference."
  }
}
```

**`zero-shot-classification`**

```json
{
  "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
  "parameters": {
    "candidate_labels": ["refund", "legal", "faq"]
  }
}
```

**`table-question-answering`**

```json
{
  "inputs": {
    "query": "How many stars does the transformers repository have?",
    "table": {
      "Repository": ["Transformers", "Datasets", "Tokenizers"],
      "Stars": ["36542", "4512", "3934"],
      "Contributors": ["651", "77", "34"],
      "Programming language": ["Python", "Python", "Rust, Python and NodeJS"]
    }
  }
}
```

## Setup & Installation

Before you can deploy a 🤗 Transformers model to Amazon SageMaker you need to sign up for an AWS account. If you do not have an AWS account yet learn more [here](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).

After you complete these tasks you can get started using either [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html), [SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html), or a local environment. To deploy from your local machine you need to configure the right [IAM permission](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

Upgrade to the latest `sagemaker` version.

```bash
pip install sagemaker --upgrade
```

**SageMaker environment**

_Note: The execution role is intended to be available only when running a notebook within SageMaker. If you run `get_execution_role` in a notebook not on SageMaker, expect a "region" error._

```python
import sagemaker
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
```

**Local environment**

```python
import sagemaker
import boto3

iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName='role-name-of-your-iam-role-with-right-permissions')['Role']['Arn']
sess = sagemaker.Session()
```

---

## Deploy for inference a 🤗 Transformers model trained in SageMaker

<iframe width="700" height="394" src="https://www.youtube.com/embed/pfBGgSGnYLs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

There are two ways to deploy your Hugging Face model trained in SageMaker. You can either deploy it after your training is finished, or you can deploy it later, using the `model_data` pointing to your saved model on S3.

- [Deploy a Hugging Face Transformer model from S3 to SageMaker for inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/10_deploy_model_from_s3/deploy_transformer_model_from_s3.ipynb)


### Deploy the model directly after training

If you deploy your model directly after training, you need to ensure that all required files are saved in your training script, including the Tokenizer and the Model.

```python
from sagemaker.huggingface import HuggingFace

############ pseudo code start ############

# create HuggingFace estimator for running training
huggingface_estimator = HuggingFace(....)

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(...)

############ pseudo code end ############

# deploy model to SageMaker Inference
predictor = hf_estimator.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")

# example request, you always need to define "inputs"
data = {
   "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
}
# request
predictor.predict(data)
```

After we run our request we can delete the endpoint again with.

```python
# delete endpoint
predictor.delete_endpoint()
```

### Deploy the model using `model_data`

If you've already trained your model and want to deploy it at some later time, you can use the `model_data` argument to specify the location of your tokenizer and model weights.

```python
from sagemaker.huggingface.model import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data="s3://models/my-bert-model/model.tar.gz",  # path to your trained sagemaker model
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.6", # transformers version used
   pytorch_version="1.7", # pytorch version used
   py_version='py36', # python version used
)
# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge"
)

# example request, you always need to define "inputs"
data = {
   "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
}

# request
predictor.predict(data)
```

After we run our request, we can delete the endpoint again with:

```python
# delete endpoint
predictor.delete_endpoint()
```

---

## Deploy for inference one of the 10,000+ 🤗 Transformers models available in the 🤗 Hub

<iframe width="700" height="394" src="https://www.youtube.com/embed/l9QZuazbzWM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- [Deploy one of the 10 000+ Hugging Face Transformers to Amazon SageMaker for Inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb)

To deploy a model directly from the 🤗 Hub to SageMaker, we need to define 2 environment variables when creating the HuggingFaceModel. We need to define:

- `HF_MODEL_ID`: defines the model id, which will be automatically loaded from[ huggingface.co/models](http://huggingface.co/models) when creating or SageMaker Endpoint. The 🤗 Hub provides 10,000+ models, all available through this environment variable.
- `HF_TASK`: defines the task for the 🤗 Transformers pipeline used. A full list of tasks can be found[ here](https://huggingface.co/transformers/main_classes/pipelines.html).

```python
from sagemaker.huggingface.model import HuggingFaceModel

# Hub Model configuration. <https://huggingface.co/models>
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad', # model_id from hf.co/models
  'HF_TASK':'question-answering' # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub, # configuration for loading model from Hub
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.6", # transformers version used
   pytorch_version="1.7", # pytorch version used
   py_version='py36', # python version used
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge"
)

# example request, you always need to define "inputs"
data = {
"inputs": {
	"question": "What is used for inference?",
	"context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference."
	}
}

# request
predictor.predict(data)
```

After we run our request, we can delete the endpoint again with.

```python
# delete endpoint
predictor.delete_endpoint()
```

---

## Advanced Features

### Environment variables

The SageMaker Hugging Face Inference Toolkit implements various additional environment variables to simplify your deployment experience. A full list of environment variables is given below.

#### `HF_TASK`

The `HF_TASK` environment variable defines the task for the 🤗 Transformers pipeline used . A full list of tasks can be find [here](https://huggingface.co/transformers/main_classes/pipelines.html).

```bash
HF_TASK="question-answering"
```

#### `HF_MODEL_ID`

The `HF_MODEL_ID` environment variable defines the model id, which will be automatically loaded from [huggingface.co/models](https://huggingface.co/models) when creating or SageMaker Endpoint. The 🤗 Hub provides 10,000+ models, all available through this environment variable.

```bash
HF_MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english"
```

#### `HF_MODEL_REVISION`

The `HF_MODEL_REVISION` is an extension to `HF_MODEL_ID` and allows you to define/pin a revision of the model to make sure you always load the same model on your SageMaker Endpoint.

```bash
HF_MODEL_REVISION="03b4d196c19d0a73c7e0322684e97db1ec397613"
```

#### `HF_API_TOKEN`

The `HF_API_TOKEN` environment variable defines your Hugging Face authorization token. The `HF_API_TOKEN` is used as a HTTP bearer authorization for remote files, like private models. You can find your token at your [settings page](https://huggingface.co/settings/token).

```bash
HF_API_TOKEN="api_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

### Creating a Model artifact `model.tar.gz` for deployment 

As shown in [Deploy the model using `model_data`](#deploy-the-model-using-model_data) you can either deploy your model directly after training or creating a `model.tar.gz` later and using it for deployment.
The `model.tar.gz` contains all required files to run your model including your model file either `pytorch_model.bin`, `tf_model.h5`, `tokenizer.json` , `tokenizer_config.json` etc. All model artifacts need to be directly in the archive without folder hierarchy.

examples for PyTorch:

```bash
model.tar.gz/
|- pytroch_model.bin
|- vocab.txt
|- tokenizer_config.json
|- config.json
|- special_tokens_map.json
```

Steps how to create a `model.tar.gz` from a model of hf.co/models


1. Download the model
```bash
git lfs install
git clone https://huggingface.co/{repository}
```

2. Create a tar file
```bash
cd {repository}
tar zcvf model.tar.gz *
```

3. Upload `model.tar.gz` to s3
```bash
aws s3 cp model.tar.gz <s3://{my-s3-path}>
```

After that you can use the S3 uri as `model_data`. 


### User defined code/modules


The Hugging Face Inference Toolkit allows user to override the default methods of the `HuggingFaceHandlerService`. Therefor the need to create a named `code/` with a `inference.py` file in it. 
For example:  

```bash
model.tar.gz/
|- pytroch_model.bin
|- ....
|- code/
  |- inference.py
  |- requirements.txt 
```

In this example, `pytroch_model.bin` is the model file saved from training, `inference.py` is the custom inference module, and `requirements.txt` is a requirements file to add additional dependencies.
The custom module can override the following methods:  

* `model_fn(model_dir)`: Overrides the default method for loading the model, the return value `model` will be used in the `predict()` for predicitions. It receives argument the `model_dir`, the path to your unzipped `model.tar.gz`.
* `transform_fn(model, data, content_type, accept_type)`: Overrides the default transform function with custom implementation. Customers using this would have to implement `preprocess`, `predict` and `postprocess` steps in the `transform_fn`. **NOTE: This method can't be combined with `input_fn`, `predict_fn` or `output_fn` mentioned below.** 
* `input_fn(input_data, content_type)`: Overrides the default method for prerprocessing, the return value `data` will be used in the `predict()` method for predicitions. The input is `input_data`, the raw body of your request and `content_type`, the content type form the request Header.
* `predict_fn(processed_data, model)`: Overrides the default method for predictions, the return value `predictions` will be used in the `postprocess()` method. The input is `processed_data`, the result of the `preprocess()` method.
* `output_fn(prediction, accept)`: Overrides the default method for postprocessing, the return value `result` will be the respond of your request(e.g.`JSON`). The inputs are `predictions`, the result of the `predict()` method and `accept` the return accept type from the HTTP Request, e.g. `application/json`



Example of an `inference.py` with `model_fn`, `input_fn`, `predict_fn` & `output_fn`:  

```python
def model_fn(model_dir):
    return "model"


def input_fn(data, content_type):
    return "data"


def predict_fn(data, model):
    return "output"


def output_fn(prediction, accept):
    return prediction
```

Example of an `inference.py` with `model_fn` & `transform_fn`:   

```python
def model_fn(model_dir):
    return "loading model"


def transform_fn(model, input_data, content_type, accept):
    return f"output"
```


### Batch Transform

After you train a model, you can use [Amazon SageMaker Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html) to perform inferences with the model. In Batch Transform you provide your inference data as a S3 uri and SageMaker will care of downloading it, running the prediction and uploading the results afterwards to S3 again. You can find more documentation for Batch Transform [here](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)

If you trained the model using the `HuggingFace` estimator, you can invoke `transformer()` method to create a transform job for a model based on the training job.

```python
batch_job = huggingface_estimator.transformer(
    instance_count=1,
    instance_type='ml.c5.2xlarge',
    strategy='SingleRecord')


batch_job.transform(
    data='s3://s3-uri-to-batch-data',
    content_type='application/json',    
    split_type='Line')
```

For more details about what can be specified here, see [API docs](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform).

If you want to run your Batch Transform Job later or with a model from hf.co/models you can do this by creating a `HuggingFaceModel` instance and then using the `transformer()` method.

```python
from sagemaker.huggingface.model import HuggingFaceModel

# Hub Model configuration. <https://huggingface.co/models>
hub = {
	'HF_MODEL_ID':'distilbert-base-uncased-finetuned-sst-2-english',
	'HF_TASK':'text-classification'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub, # configuration for loading model from Hub
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.6", # transformers version used
   pytorch_version="1.7", # pytorch version used
       py_version='py36', # python version used
)

# create Transformer to run our batch job
batch_job = huggingface_model.transformer(
    instance_count=1,
    instance_type='ml.c5.2xlarge',
    strategy='SingleRecord')

# starts batch transform job and uses s3 data as input
batch_job.transform(
    data='s3://sagemaker-s3-demo-test/samples/input.jsonl',
    content_type='application/json',    
    split_type='Line')
```

The `input.jsonl` looks like this.

```jsonl
{"inputs":"this movie is terrible"}
{"inputs":"this movie is amazing"}
{"inputs":"SageMaker is pretty cool"}
{"inputs":"SageMaker is pretty cool"}
{"inputs":"this movie is terrible"}
{"inputs":"this movie is amazing"}
```


## Additional Resources

- [Announcement Blog Post](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face)

- [Deploy Hugging Face models easily with Amazon SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker)

- [AWS and Hugging Face collaborate to simplify and accelerate adoption of natural language processing](https://aws.amazon.com/blogs/machine-learning/aws-and-hugging-face-collaborate-to-simplify-and-accelerate-adoption-of-natural-language-processing-models/)

- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
