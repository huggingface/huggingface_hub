---
title: Frequently Asked Questions
---

<h1>Frequently Asked Questions 🎯</h1>

## Q: What are Deep Learning Containers?

A: Deep Learning Containers (DLCs) are Docker images pre-installed with deep learning frameworks and libraries (e.g. transformers, datasets, tokenizers) to make it easy to train models by letting you skip the complicated process of building and optimizing your environments from scratch.

## Q: Why should I use the Hugging Face Deep Learning Containers?

A: The DLCs are fully tested, maintained, optimized deep learning environments that require no installation, configuration, or maintenance. In particular, the Hugging Face inference DLC comes with a pre-written serving stack, which drastically lowers the technical bar of DL serving.

## Q: Do I have to use the SageMaker Python SDK to use the Hugging Face Deep Learning Containers?

A: You can use the HF DLC without the SageMaker Python SDK and launch SageMaker Training jobs with other SDKs, such as the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-training-job.html) or [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job). The DLCs are also available through Amazon ECR and can be pulled and used in any environment of choice.


## Q: Why should I use SageMaker Training to train Hugging Face models?

A: SageMaker Training provides numerous benefits that will boost your productivity with Hugging Face : (1) first it is cost-effective: the training instances live only for the duration of your job and are paid per second. No risk anymore to leave GPU instances up all night: the training cluster stops right at the end of your job! It also supports EC2 Spot capacity, which enables up to 90% cost reduction. (2) SageMaker also comes with a lot of built-in automation that facilitates teamwork and MLOps: training metadata and logs are automatically persisted to a serverless managed metastore, and I/O with S3 (for datasets, checkpoints and model artifacts) is fully managed. Finally, SageMaker also allows to drastically scale up and out: you can launch multiple training jobs in parallel, but also launch large-scale distributed training jobs

## Q: Once I've trained my model with Amazon SageMaker, can I use it with 🤗/Transformers ?

A: Yes, you can download your trained model from S3 and directly use it with transformers or upload it to the [Hugging Face Model Hub](https://huggingface.co/models).

## Q: How is my data and code secured by Amazon SageMaker?

A: Amazon SageMaker provides numerous security mechanisms including [encryption at rest](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-at-rest-nbi.html) and [in transit](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-in-transit.html), [Virtual Private Cloud (VPC) connectivity](https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html) and [Identity and Access Management (IAM)](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html). To learn more about security in the AWS cloud and with Amazon SageMaker, you can visit [Security in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html) and [AWS Cloud Security](https://docs.aws.amazon.com/sagemaker/latest/dg/security_iam_service-with-iam.html).

## Q: Is this available in my region?

A: For a list of the supported regions, please visit the [AWS region table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) for all AWS global infrastructure.

## Q: Do I need to pay for a license from Hugging Face to use the DLCs?

A: No - the Hugging Face DLCs are open source and licensed under Apache 2.0.

## Q: How can I run inference on my trained models?

A: You have multiple options to run inference on your trained models. One option is to use [the SageMaker Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) to run inference in Amazon SageMaker. This Inference Toolkit provides default pre-processing, predict and postprocessing for certain 🤗 Transformers models and tasks. It utilizes the SageMaker Inference Toolkit for starting up the model server, which is responsible for handling inference requests. Another great option is to use Hugging Face [Accelerated Inference-API](https://api-inference.huggingface.co/docs/python/html/index.html) hosted service: start by [uploading the trained models to your Hugging Face account](https://huggingface.co/new) to deploy them publicly, or privately.
## Q: Which models can I deploy for Inference?

A: You can deploy

- any 🤗 Transformers model trained in Amazon SageMaker, or other compatible platforms and that can accomodate the SageMaker Hosting design
- any of the 10 000+ publicly available Transformer models from the Hugging Face [Model Hub](https://huggingface.co/models), or
- your private models hosted in your Hugging Face premium account!

## Q: Which pipelines, tasks are supported by the Inference Toolkit?

A: The Inference Toolkit and DLC support any of the `transformers` `pipelines`. You can find the full list[ here](https://huggingface.co/transformers/main_classes/pipelines.html)

## Q: Do I have to use the transformers pipelines when hosting SageMaker endpoints?  

A: No, you can also write your custom inference code to serve your own models and logic, documented here.  

## Q: Do you offer premium support or support SLAs for this solution?

A: AWS Technical Support tiers are available from AWS and cover development and production issues for AWS products and services - please refer to AWS Support for specifics and scope.

If you have questions which the Hugging Face community can help answer and/or benefit from, please [post them in the Hugging Face forum](https://discuss.huggingface.co/c/sagemaker/17).

If you need premium support from the Hugging Face team to accelerate your NLP roadmap, our Expert Acceleration Program offers direct guidance from our open source, science and ML Engineering team - [contact us to learn more](mailto:api-enterprise@huggingface.co).

## Q: What are you planning next through this partnership?

A: Our common goal is to democratize state of the art Machine Learning. We will continue to innovate to make it easier for researchers, data scientists and ML practitioners to manage, train and run state of the art models. If you have feature requests for integration in AWS with Hugging Face, please [let us know in the Hugging Face community forum](https://discuss.huggingface.co/c/sagemaker/17).

## Q: I use Hugging Face with Azure Machine Learning or Google Cloud Platform, what does this partnership mean for me?

A: A foundational goal for Hugging Face is to make the latest AI accessible to as many people as possible, whichever framework or development environment they work in. While we are focusing integration efforts with Amazon Web Services as our Preferred Cloud Provider, we will continue to work hard to serve all Hugging Face users and customers, no matter what compute environment they run on.
