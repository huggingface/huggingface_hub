---
title: Hugging Face on Amazon SageMaker
---

<h1>Hugging Face on Amazon SageMaker</h1>

![cover](/docs/assets/sagemaker/cover.png)


Earlier this year [we announced a strategic collaboration with Amazon](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face) to make it easier for companies to use Hugging Face in SageMaker, and ship cutting-edge Machine Learning features faster. We introduced new Hugging Face Deep Learning Containers (DLCs) to[ train Hugging Face Transformer models in Amazon SageMaker](https://huggingface.co/transformers/sagemaker.html#getting-started-train-a-transformers-model).


## Features & Benefits 🔥

### One Command is All you Need

With the new Hugging Face Deep Learning Containers available in Amazon SageMaker, training <!-- and deploying -->
cutting-edge Transformers-based NLP models has never been simpler. There are variants specially optimized for TensorFlow and PyTorch, for single-GPU, single-node multi-GPU and multi-node clusters.

### Accelerating Machine Learning from Science to Production

In addition to Hugging Face DLCs, we created a first-class Hugging Face extension to the SageMaker Python-sdk to accelerate data science teams, reducing the time required to set up and run experiments from days to minutes.

You can use the Hugging Face DLCs with the Automatic Model Tuning capability of Amazon SageMaker, in order to automatically optimize your training hyperparameters and quickly increase the accuracy of your models.

You can deploy your trained models for inference with just one more line of code, or select any of the 10,000+ publicly available models from the Model Hub, and deploy them with Amazon SageMaker.


Thanks to the SageMaker Studio web-based Integrated Development Environment (IDE), you can easily track and compare your experiments and your training artifacts.

### Built-in Performance

With the Hugging Face DLCs, SageMaker customers will benefit from built-in performance optimizations for PyTorch or TensorFlow, to train NLP models faster, and with the flexibility to choose the training<!-- inference --> infrastructure with the best price/performance ratio for your workload.

The Hugging Face Training DLCs are fully integrated with the SageMaker distributed training libraries, to train models faster than was ever possible before, using the latest generation of instances available on Amazon EC2.

The Hugging Face Inference DLCs provide you with production-ready endpoints that scale easily within your AWS environment, with built-in monitoring and a ton of enterprise features. 


---

## Resources, Documentation & Samples 📄

Below you can find all the important resources to all published blog posts, videos, documentation, and sample Notebooks/scripts.

### Blog/Video

- [AWS: Embracing natural language processing with Hugging Face](https://aws.amazon.com/de/blogs/opensource/embracing-natural-language-processing-with-hugging-face/)
- [Deploy Hugging Face models easily with Amazon SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker)
- [AWS and Hugging Face collaborate to simplify and accelerate adoption of natural language processing models](https://aws.amazon.com/blogs/machine-learning/aws-and-hugging-face-collaborate-to-simplify-and-accelerate-adoption-of-natural-language-processing-models/)
- [Walkthrough: End-to-End Text Classification](https://youtu.be/ok3hetb42gU)
- [Working with Hugging Face models on Amazon SageMaker](https://youtu.be/leyrCgLAGjMn)
- [Distributed Training: Train BART/T5 for Summarization using 🤗 Transformers and Amazon SageMaker](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq)
- [Deploy a Hugging Face Transformers Model from S3 to Amazon SageMaker](https://youtu.be/pfBGgSGnYLs)
- [Deploy a Hugging Face Transformers Model from the Model Hub to Amazon SageMaker](https://youtu.be/l9QZuazbzWM)

### Documentation

- [Run training on Amazon SageMaker](/docs/sagemaker/train)
- [Deploy models to Amazon SageMaker](/docs/sagemaker/inference)
- [Frequently Asked Questions](/docs/sagemaker/faq)
- [Amazon SageMaker documentation for Hugging Face](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)
- [Python SDK SageMaker documentation for Hugging Face](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html)
- [Deep Learning Container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers)
- [SageMaker's Distributed Data Parallel Library](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html)
- [SageMaker's Distributed Model Parallel Library](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html)

### Sample Notebooks

- [all Notebooks](https://github.com/huggingface/notebooks/tree/master/sagemaker)
- [Getting Started Pytorch](https://github.com/huggingface/notebooks/blob/master/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb)
- [Getting Started Tensorflow](https://github.com/huggingface/notebooks/blob/master/sagemaker/02_getting_started_tensorflow/sagemaker-notebook.ipynb)
- [Distributed Training Data Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/03_distributed_training_data_parallelism/sagemaker-notebook.ipynb)
- [Distributed Training Model Parallelism](https://github.com/huggingface/notebooks/blob/master/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb)
- [Spot Instances and continue training](https://github.com/huggingface/notebooks/blob/master/sagemaker/05_spot_instances/sagemaker-notebook.ipynb)
- [SageMaker Metrics](https://github.com/huggingface/notebooks/blob/master/sagemaker/06_sagemaker_metrics/sagemaker-notebook.ipynb)
- [Distributed Training Data Parallelism Tensorflow](https://github.com/huggingface/notebooks/blob/master/sagemaker/07_tensorflow_distributed_training_data_parallelism/sagemaker-notebook.ipynb)
- [Distributed Training Summarization](https://github.com/huggingface/notebooks/blob/master/sagemaker/08_distributed_summarization_bart_t5/sagemaker-notebook.ipynb)
- [Image Classification with Vision Transformer](https://github.com/huggingface/notebooks/blob/master/sagemaker/09_image_classification_vision_transformer/sagemaker-notebook.ipynb)
- [Deploy one of the 10 000+ Hugging Face Transformers to Amazon SageMaker for Inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/11_deploy_model_from_hf_hub/deploy_transformer_model_from_hf_hub.ipynb)
- [Deploy a Hugging Face Transformer model from S3 to SageMaker for inference](https://github.com/huggingface/notebooks/blob/master/sagemaker/10_deploy_model_from_s3/deploy_transformer_model_from_s3.ipynb)

---

## Deep Learning Container (DLC) overview

The Deep Learning Container are available everywhere Amazon SageMaker is available. You can see the [AWS region table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) for all AWS global infrastructure. To get an detailed overview of all included packages look [here in the release notes](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html).

| 🤗 Transformers version | 🤗 Datasets version | PyTorch/TensorFlow version | type     | device | Python Version | Example `image_uri`                                                                                                               |
| ----------------------- | ------------------- | -------------------------- | -------- | ------ | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 4.4.2                   | 1.5.0               | PyTorch 1.6.0              | training | GPU    | 3.6            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04`    |
| 4.4.2                   | 1.5.0               | TensorFlow 2.4.1           | training | GPU    | 3.7            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-tensorflow-training:2.4.1-transformers4.4.2-gpu-py37-cu110-ubuntu18.04` |
| 4.5.0                   | 1.5.0               | PyTorch 1.6.0              | training | GPU    | 3.6            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04`    |
| 4.5.0                   | 1.5.0               | TensorFlow 2.4.1           | training | GPU    | 3.7            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-tensorflow-training:2.4.1-transformers4.5.0-gpu-py37-cu110-ubuntu18.04` |
| 4.6.1                   | 1.6.2               | PyTorch 1.6.0              | training | GPU    | 3.6            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.5.0-gpu-py36-cu110-ubuntu18.04`    |
| 4.6.1                   | 1.6.2               | PyTorch 1.7.1               | training | GPU    | 3.6            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04` |
| 4.6.1                   | 1.6.2               | TensorFlow 2.4.1           | training | GPU    | 3.7            | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-tensorflow-training:2.4.1-transformers4.6.1-gpu-py37-cu110-ubuntu18.04` |
| 4.6.1                   | 1.6.2               | PyTorch 1.7.1               | inference | CPU    | 3.6            | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04` |
| 4.6.1                   | 1.6.2               | PyTorch 1.7.1               | inference | GPU    | 3.6            | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04` |
| 4.6.1                   | 1.6.2               | TensorFlow 2.4.1           | inference | CPU    | 3.7            | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.4.1-transformers4.6.1-cpu-py37-ubuntu18.04` |
| 4.6.1                   | 1.6.2               | TensorFlow 2.4.1           | inference | GPU    | 3.7            | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.4.1-transformers4.6.1-gpu-py37-cu110-ubuntu18.04` |

---
