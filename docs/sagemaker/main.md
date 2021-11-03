---
title: Hugging Face on Amazon SageMaker
---

<h1>Hugging Face on Amazon SageMaker</h1>

![cover](/docs/assets/sagemaker/cover.png)

## Deep Learning Containers

Deep Learning Containers (DLCs) are Docker images pre-installed with deep learning frameworks and libraries such as 🤗 Transformers, 🤗 Datasets, and 🤗 Tokenizers. The DLCs allow you to start training models immediately, skipping the complicated process of building and optimizing your training environments from scratch. Our DLCs are thoroughly tested and optimized for deep learning environments, requiring no configuration or maintenance on your part. In particular, the Hugging Face Inference DLC comes with a pre-written serving stack which drastically lowers the technical bar of deep learning serving.

Our DLCs are available everywhere [Amazon SageMaker](https://aws.amazon.com/sagemaker/) is [available](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/). While it is possible to use the DLCs without the SageMaker Python SDK, there are many advantages to using SageMaker to train your model:

- Cost-effective: Training instances are only live for the duration of your job. Once your job is complete, the training cluster stops, and you won't be billed anymore. SageMaker also supports [Spot instances]((https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)), which can reduce costs up to 90%.
- Built-in automation: SageMaker automatically stores training metadata and logs in a serverless managed metastore and fully manages I/O operations with S3 for your datasets, checkpoints, and model artifacts.
- Multiple security mechanisms: SageMaker offers [encryption at rest](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-at-rest-nbi.html), [in transit](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-in-transit.html), [Virtual Private Cloud](https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html) connectivity, and [Identity and Access Management](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html) to secure your data and code.

Hugging Face DLCs are open source and licensed under Apache 2.0. Feel free to reach out on our [community forum](https://discuss.huggingface.co/c/sagemaker/17) if you have any questions. For premium support, our [Expert Acceleration Program](https://huggingface.co/support) gives you direct dedicated support from our team.

## Features & benefits 🔥

Hugging Face Deep DLCs make it easier than ever to train Transformer models in SageMaker. Here is why you should consider using Hugging Face DLCs to train and deploy your next machine learning models:

**One command is all you need**

With the new Hugging Face DLCs, train cutting-edge Transformers-based NLP models in a single line of code. Choose from multiple DLC variants, each one optimized for TensorFlow and PyTorch, single-GPU, single-node multi-GPU, and multi-node clusters.

**Accelerate machine learning from science to production**

In addition to Hugging Face DLCs, we created a first-class Hugging Face extension for the SageMaker Python SDK to accelerate data science teams, reducing the time required to set up and run experiments from days to minutes.

You can use the Hugging Face DLCs with SageMaker's automatic model tuning to optimize your training hyperparameters and increase the accuracy of your models.

Deploy your trained models for inference with just one more line of code or select any of the 10,000+ publicly available models from the [model Hub](https://huggingface.co/models) and deploy them with SageMaker.

Easily track and compare your experiments and training artifacts in SageMaker Studio's web-based integrated development environment (IDE).

**Built-in performance**

Hugging Face DLCs feature built-in performance optimizations for PyTorch and TensorFlow to train NLP models faster. The DLCs also give you the flexibility to choose a training infrastructure that best aligns with the price/performance ratio for your workload.

The Hugging Face Training DLCs are fully integrated with SageMaker distributed training libraries to train models faster than ever, using the latest generation of instances available on Amazon Elastic Compute Cloud.

Hugging Face Inference DLCs provide you with production-ready endpoints that scale quickly with your AWS environment, built-in monitoring, and a ton of enterprise features. 

---

## Resources, Documentation & Samples 📄

Take a look at our published blog posts, videos, documentation, sample notebooks and scripts for additional help and more context about Hugging Face DLCs on SageMaker.

### Blogs and videos

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

### Sample notebooks

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