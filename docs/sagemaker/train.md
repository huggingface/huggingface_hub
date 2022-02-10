---
title: Run training on Amazon SageMaker
---

<h1>Run training on Amazon SageMaker</h1>

<iframe width="700" height="394" src="https://www.youtube.com/embed/ok3hetb42gU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


This guide will show you how to train a 🤗 Transformers model with the `HuggingFace` SageMaker Python SDK. Learn how to:

- [Install and setup your training environment](#installation-and-setup).
- [Prepare a training script](#prepare-a-transformers-fine-tuning-script).
- [Create a Hugging Face Estimator](#create-a-hugging-face-estimator).
- [Run training with the `fit` method](#execute-training).
- [Access your trained model](#access-trained-model).
- [Perform distributed training](#distributed-training).
- [Create a spot instance](#spot-instances).
- [Load a training script from a GitHub repository](#git-repository).
- [Collect training metrics](#sagemaker-metrics).

## Installation and setup

Before you can train a 🤗 Transformers model with SageMaker, you need to sign up for an AWS account. If you don't have an AWS account yet, learn more [here](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).

Once you have an AWS account, get started using one of the following:

- [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html)
- [SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html)
- Local environment

To start training locally, you need to setup an appropriate [IAM role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

Upgrade to the latest `sagemaker` version:

```bash
pip install sagemaker --upgrade
```

**SageMaker environment**

Setup your SageMaker environment as shown below:

```python
import sagemaker
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
```

_Note: The execution role is only available when running a notebook within SageMaker. If you run `get_execution_role` in a notebook not on SageMaker, expect a `region` error._

**Local environment**

Setup your local environment as shown below:

```python
import sagemaker
import boto3

iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName='role-name-of-your-iam-role-with-right-permissions')['Role']['Arn']
sess = sagemaker.Session()
```

## Prepare a 🤗 Transformers fine-tuning script

Our training script is very similar to a training script you might run outside of SageMaker. However, you can access useful properties about the training environment through various environment variables (see [here](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md) for a complete list), such as:

- `SM_MODEL_DIR`: A string representing the path to which the training job writes the model artifacts. After training, artifacts in this directory are uploaded to S3 for model hosting. `SM_MODEL_DIR` is always set to `/opt/ml/model`.

- `SM_NUM_GPUS`: An integer representing the number of GPUs available to the host.

- `SM_CHANNEL_XXXX:` A string representing the path to the directory that contains the input data for the specified channel. For example, when you specify `train` and `test` in the Hugging Face Estimator `fit` method, the environment variables are set to `SM_CHANNEL_TRAIN` and `SM_CHANNEL_TEST`.

The `hyperparameters` defined in the [Hugging Face Estimator](#create-an-huggingface-estimator) are passed as named arguments and processed by `ArgumentParser()`.

```python
import transformers
import datasets
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--model_name_or_path", type=str)

    # data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
```

_Note that SageMaker doesn’t support argparse actions. For example, if you want to use a boolean hyperparameter, specify `type` as `bool` in your script and provide an explicit `True` or `False` value._

Look [here](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/scripts/train.py) for a complete example of a 🤗 Transformers training script.

## Training Output Management

If `output_dir` in the `TrainingArguments` is set to '/opt/ml/model' the Trainer saves all training artifacts, including logs, checkpoints, and models. Amazon SageMaker archives the whole '/opt/ml/model' directory as `model.tar.gz` and uploads it at the end of the training job to Amazon S3. Depending on your Hyperparameters and `TrainingArguments` this could lead to a large artifact (> 5GB), which can slow down deployment for Amazon SageMaker Inference. 
You can control how checkpoints, logs, and artifacts are saved by customization the [TrainingArguments](https://huggingface.co/docs/transformers/master/en/main_classes/trainer#transformers.TrainingArguments). For example by providing `save_total_limit` as `TrainingArgument` you can control the limit of the total amount of checkpoints. Deletes the older checkpoints in `output_dir` if new ones are saved and the maximum limit is reached.

In addition to the options already mentioned above, there is another option to save the training artifacts during the training session. Amazon SageMaker supports [Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html), which allows you to continuously save your artifacts during training to Amazon S3 rather than at the end of your training. To enable [Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) you need to provide the `checkpoint_s3_uri` parameter pointing to an Amazon S3 location in the `HuggingFace` estimator and set `output_dir` to `/opt/ml/checkpoints`. 
_Note: If you set `output_dir` to `/opt/ml/checkpoints` make sure to call `trainer.save_model("/opt/ml/model")` or model.save_pretrained("/opt/ml/model")/`tokenizer.save_pretrained("/opt/ml/model")` at the end of your training to be able to deploy your model seamlessly to Amazon SageMaker for Inference._

## Create a Hugging Face Estimator

Run 🤗 Transformers training scripts on SageMaker by creating a [Hugging Face Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#huggingface-estimator). The Estimator handles end-to-end SageMaker training. There are several parameters you should define in the Estimator:

1. `entry_point` specifies which fine-tuning script to use.
2. `instance_type` specifies an Amazon instance to launch. Refer [here](https://aws.amazon.com/sagemaker/pricing/) for a complete list of instance types.
3. `hyperparameters` specifies training hyperparameters. View additional available hyperparameters [here](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/scripts/train.py).

The following code sample shows how to train with a custom script `train.py` with three hyperparameters (`epochs`, `per_device_train_batch_size`, and `model_name_or_path`):

```python
from sagemaker.huggingface import HuggingFace


# hyperparameters which are passed to the training job
hyperparameters={'epochs': 1,
                 'per_device_train_batch_size': 32,
                 'model_name_or_path': 'distilbert-base-uncased'
                 }

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters = hyperparameters
)
```

If you are running a `TrainingJob` locally, define `instance_type='local'` or `instance_type='local-gpu'` for GPU usage. Note that this will not work with SageMaker Studio.

## Execute training

Start your `TrainingJob` by calling `fit` on a Hugging Face Estimator. Specify your input training data in `fit`. The input training data can be a:

- S3 URI such as `s3://my-bucket/my-training-data`.
- `FileSystemInput` for Amazon Elastic File System or FSx for Lustre. See [here](https://sagemaker.readthedocs.io/en/stable/overview.html?highlight=FileSystemInput#use-file-systems-as-training-inputs) for more details about using these file systems as input.

Call `fit` to begin training:

```python
huggingface_estimator.fit(
  {'train': 's3://sagemaker-us-east-1-558105141721/samples/datasets/imdb/train',
   'test': 's3://sagemaker-us-east-1-558105141721/samples/datasets/imdb/test'}
)
```

SageMaker starts and manages all the required EC2 instances and initiates the `TrainingJob` by running:

```bash
/opt/conda/bin/python train.py --epochs 1 --model_name_or_path distilbert-base-uncased --per_device_train_batch_size 32
```

## Access trained model

Once training is complete, you can access your model through the [AWS console](https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin) or download it directly from S3.

```python
from sagemaker.s3 import S3Downloader

S3Downloader.download(
    s3_uri=huggingface_estimator.model_data, # S3 URI where the trained model is located
    local_path='.',                          # local path where *.targ.gz is saved
    sagemaker_session=sess                   # SageMaker session used for training the model
)
```

## Distributed training

SageMaker provides two strategies for distributed training: data parallelism and model parallelism. Data parallelism splits a training set across several GPUs, while model parallelism splits a model across several GPUs.

### Data parallelism

The Hugging Face [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) supports SageMaker's data parallelism library. If your training script uses the Trainer API, you only need to define the distribution parameter in the Hugging Face Estimator:

```python
# configuration for running training on smdistributed data parallel
distribution = {'smdistributed':{'dataparallel':{ 'enabled': True }}}

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3dn.24xlarge',
        instance_count=2,
        role=role,
        transformers_version='4.4.2',
        pytorch_version='1.6.0',
        py_version='py36',
        hyperparameters = hyperparameters
        distribution = distribution
)
```

📓 Open the [notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb) for an example of how to run the data parallelism library with TensorFlow.

### Model parallelism

The Hugging Face [Trainer] also supports SageMaker's model parallelism library. If your training script uses the Trainer API, you only need to define the distribution parameter in the Hugging Face Estimator (see [here](https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel_general.html?highlight=modelparallel#required-sagemaker-python-sdk-parameters) for more detailed information about using model parallelism):

```python
# configuration for running training on smdistributed model parallel
mpi_options = {
    "enabled" : True,
    "processes_per_host" : 8
}

smp_options = {
    "enabled":True,
    "parameters": {
        "microbatches": 4,
        "placement_strategy": "spread",
        "pipeline": "interleaved",
        "optimize": "speed",
        "partitions": 4,
        "ddp": True,
    }
}

distribution={
    "smdistributed": {"modelparallel": smp_options},
    "mpi": mpi_options
}

 # create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3dn.24xlarge',
        instance_count=2,
        role=role,
        transformers_version='4.4.2',
        pytorch_version='1.6.0',
        py_version='py36',
        hyperparameters = hyperparameters,
        distribution = distribution
)
```

📓 Open the [notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb) for an example of how to run the model parallelism library.

## Spot instances

The Hugging Face extension for the SageMaker Python SDK means we can benefit from [fully-managed EC2 spot instances](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html). This can help you save up to 90% of training costs!

_Note: Unless your training job completes quickly, we recommend you use [checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) with managed spot training. In this case, you need to define the `checkpoint_s3_uri`._

Set `use_spot_instances=True` and define your `max_wait` and `max_run` time in the Estimator to use spot instances:

```python
# hyperparameters which are passed to the training job
hyperparameters={'epochs': 1,
                 'train_batch_size': 32,
                 'model_name':'distilbert-base-uncased',
                 'output_dir':'/opt/ml/checkpoints'
                 }

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
	    checkpoint_s3_uri=f's3://{sess.default_bucket()}/checkpoints'
        use_spot_instances=True,
        # max_wait should be equal to or greater than max_run in seconds
        max_wait=3600,
        max_run=1000,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters = hyperparameters
)

# Training seconds: 874
# Billable seconds: 262
# Managed Spot Training savings: 70.0%
```

📓 Open the [notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb) for an example of how to use spot instances.

## Git repository

The Hugging Face Estimator can load a training script [stored in a GitHub repository](https://sagemaker.readthedocs.io/en/stable/overview.html#use-scripts-stored-in-a-git-repository). Provide the relative path to the training script in `entry_point` and the relative path to the directory in `source_dir`.

If you are using `git_config` to run the [🤗 Transformers example scripts](https://github.com/huggingface/transformers/tree/master/examples), you need to configure the correct `'branch'` in `transformers_version` (e.g. if you use `transformers_version='4.4.2` you have to use `'branch':'v4.4.2'`). 

_Tip: Save your model to S3 by setting `output_dir=/opt/ml/model` in the hyperparameter of your training script._

```python
# configure git settings
git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.4.2'} # v4.4.2 refers to the transformers_version you use in the estimator

 # create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='run_glue.py',
        source_dir='./examples/pytorch/text-classification',
        git_config=git_config,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        hyperparameters=hyperparameters
)
```

## SageMaker metrics

[SageMaker metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html#define-train-metrics) automatically parses training job logs for metrics and sends them to CloudWatch. If you want SageMaker to parse the logs, you must specify the metric's name and a regular expression for SageMaker to use to find the metric.

```python
# define metrics definitions
metric_definitions = [
    {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
    {"Name": "eval_accuracy", "Regex": "eval_accuracy.*=\D*(.*?)$"},
    {"Name": "eval_loss", "Regex": "eval_loss.*=\D*(.*?)$"},
]

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        metric_definitions=metric_definitions,
        hyperparameters = hyperparameters)
```

📓 Open the [notebook](https://github.com/huggingface/notebooks/blob/master/sagemaker/06_sagemaker_metrics/sagemaker-notebook.ipynb) for an example of how to capture metrics in SageMaker.
