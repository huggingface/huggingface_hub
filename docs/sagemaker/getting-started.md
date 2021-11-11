---
title: Get started
---

<h1>Train and deploy Hugging Face on Amazon SageMaker</h1>

The get started guide will show you how to quickly use Hugging Face on Amazon SageMaker. Learn how to fine-tune and deploy a pretrained 🤗 Transformers model on SageMaker for a binary text classification task.

💡 If you are new to Hugging Face, we recommend first reading the 🤗 Transformers [quick tour](https://huggingface.co/transformers/quicktour.html).

<iframe width="560" height="315" src="https://www.youtube.com/embed/pYqjCzoyWyo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

📓 Open the [notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb) to follow along!

## Installation and setup

Get started by installing the necessary Hugging Face libraries and SageMaker. You will also need to install [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available) if you don't already have it installed.

```python
pip install "sagemaker>=2.48.0" "transformers==4.6.1" "datasets[s3]==1.6.2" --upgrade
```

If you want to run this example in [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html), upgrade [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) for the 🤗 Datasets library and restart the kernel:

```python
%%capture
import IPython
!conda install -c conda-forge ipywidgets -y
IPython.Application.instance().kernel.do_shutdown(True)
```

Next, you should set up your environment: a SageMaker session and an S3 bucket. The S3 bucket will store data, models, and logs. You will need access to an [IAM execution role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) with the required permissions.

If you are planning on using SageMaker in a local environment, you need to provide the `role` yourself. Learn more about how to set this up [here](https://huggingface.co/docs/sagemaker/train#installation-and-setup).

⚠️ The execution role is only available when you run a notebook within SageMaker. If you try to run `get_execution_role` in a notebook not on SageMaker, you will get a region error.

```python
import sagemaker

sess = sagemaker.Session()
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    sagemaker_session_bucket = sess.default_bucket()

role = sagemaker.get_execution_role()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
```

## Preprocess

The 🤗 Datasets library makes it easy to download and preprocess a dataset for training. Download and tokenize the [IMDb](https://huggingface.co/datasets/imdb) dataset:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# load dataset
train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# create tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

# tokenize train and test datasets
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# set dataset format for PyTorch
train_dataset =  train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

## Upload dataset to S3 bucket

Next, upload the preprocessed dataset to your S3 session bucket with 🤗 Datasets S3 [filesystem](https://huggingface.co/docs/datasets/filesystems.html) implementation:

```python
import botocore
from datasets.filesystems import S3FileSystem

s3_prefix = 'samples/datasets/imdb'
s3 = S3FileSystem()

# save train_dataset to S3
training_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/train'
train_dataset.save_to_disk(training_input_path,fs=s3)

# save test_dataset to S3
test_input_path = f's3://{sess.default_bucket()}/{s3_prefix}/test'
test_dataset.save_to_disk(test_input_path,fs=s3)
```

## Start a training job

Create a Hugging Face Estimator to handle end-to-end SageMaker training and deployment. The most important parameters to pay attention to are:

* `entry_point` refers to the fine-tuning script which you can find [here](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/scripts/train.py).
* `instance_type` refers to the SageMaker instance that will be launched. Take a look [here](https://aws.amazon.com/sagemaker/pricing/) for a complete list of instance types.
* `hyperparameters` refers to the training hyperparameters the model will be fine-tuned with.

```python
from sagemaker.huggingface import HuggingFace

hyperparameters={
    "epochs": 1,                            # number of training epochs
    "train_batch_size": 32,                 # training batch size
    "model_name":"distilbert-base-uncased"  # name of pretrained model
}

huggingface_estimator = HuggingFace(
    entry_point="train.py",                 # fine-tuning script to use in training job
    source_dir="./scripts",                 # directory where fine-tuning script is stored
    instance_type="ml.p3.2xlarge",          # instance type
    instance_count=1,                       # number of instances
    role=role,                              # IAM role used in training job to acccess AWS resources (S3)
    transformers_version="4.6",             # Transformers version
    pytorch_version="1.7",                  # PyTorch version
    py_version="py36",                      # Python version
    hyperparameters=hyperparameters         # hyperparameters to use in training job
)
```

Begin training with one line of code:

```python
huggingface_estimator.fit({"train": training_input_path, "test": test_input_path})
```

## Deploy model

Once the training job is complete, deploy your fine-tuned model by calling `deploy()` with the number of instances and instance type:

```python
predictor = huggingface_estimator.deploy(initial_instance_count=1,"ml.g4dn.xlarge")
```

Call `predict()` on your data:

```python
sentiment_input = {"inputs": "It feels like a curtain closing...there was an elegance in the way they moved toward conclusion. No fan is going to watch and feel short-changed."}

predictor.predict(sentiment_input)
```

After running your request, delete the endpoint:

```python
predictor.delete_endpoint()
```

## What's next?

Congratulations, you've just fine-tuned and deployed a pretrained 🤗 Transformers model on SageMaker! 🎉

For your next steps, keep reading our documentation for more details about training and deployment. There are many interesting features such as [distributed training](/docs/sagemaker/train#distributed-training) and [Spot instances](/docs/sagemaker/train#spot-instances).
