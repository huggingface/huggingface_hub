---
title: Reference
---

<h1>Reference</h1>

## Deep Learning Container

Below you can find a version table of currently available Hugging Face DLCs. The table doesn't include the full `image_uri` here are two examples on how to construct those if needed.

**Manually construction the `image_uri`**

`{dlc-aws-account-id}.dkr.ecr.{region}.amazonaws.com/huggingface-{framework}-{(training | inference)}:{framework-version}-transformers{transformers-version}-{device}-{python-version}-{device-tag}`

- `dlc-aws-account-id`: The AWS account ID of the account that owns the ECR repository. You can find them in the [here](https://github.com/aws/sagemaker-python-sdk/blob/e0b9d38e1e3b48647a02af23c4be54980e53dc61/src/sagemaker/image_uri_config/huggingface.json#L21)
- `region`: The AWS region where you want to use it.
- `framework`: The framework you want to use, either `pytorch` or `tensorflow`.
- `(training | inference)`: The training or inference mode.
- `framework-version`: The version of the framework you want to use.
- `transformers-version`: The version of the transformers library you want to use.
- `device`: The device you want to use, either `cpu` or `gpu`.
- `python-version`: The version of the python of the DLC.
- `device-tag`: The device tag you want to use. The device tag can include os version and cuda version

**Example 1: PyTorch Training:**
`763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04`
**Example 2: Tensorflow Inference:**
`763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.4.1-transformers4.6.1-cpu-py37-ubuntu18.04`

## Training DLC Overview

| 🤗 Transformers version | 🤗 Datasets version | PyTorch/TensorFlow version | type     | device | Python Version |
| ----------------------- | ------------------- | -------------------------- | -------- | ------ | -------------- |
| 4.4.2                   | 1.5.0               | PyTorch 1.6.0              | training | GPU    | 3.6            |
| 4.4.2                   | 1.5.0               | TensorFlow 2.4.1           | training | GPU    | 3.7            |
| 4.5.0                   | 1.5.0               | PyTorch 1.6.0              | training | GPU    | 3.6            |
| 4.5.0                   | 1.5.0               | TensorFlow 2.4.1           | training | GPU    | 3.7            |
| 4.6.1                   | 1.6.2               | PyTorch 1.6.0              | training | GPU    | 3.6            |
| 4.6.1                   | 1.6.2               | PyTorch 1.7.1              | training | GPU    | 3.6            |
| 4.6.1                   | 1.6.2               | TensorFlow 2.4.1           | training | GPU    | 3.7            |
| 4.10.2                  | 1.11.0              | PyTorch 1.8.1              | training | GPU    | 3.6            |
| 4.10.2                  | 1.11.0              | PyTorch 1.9.0              | training | GPU    | 3.8            |
| 4.10.2                  | 1.11.0              | TensorFlow 2.4.1           | training | GPU    | 3.7            |
| 4.10.2                  | 1.11.0              | TensorFlow 2.5.1           | training | GPU    | 3.7            |
| 4.11.0                  | 1.12.1              | PyTorch 1.9.0              | training | GPU    | 3.8            |
| 4.11.0                  | 1.12.1              | TensorFlow 2.5.1           | training | GPU    | 3.7            |
| 4.12.3                  | 1.15.1              | PyTorch 1.9.1              | training | GPU    | 3.8            |
| 4.12.3                  | 1.15.1              | TensorFlow 2.5.1           | training | GPU    | 3.7            |

## Inference DLC Overview

| 🤗 Transformers version | PyTorch/TensorFlow version | type      | device | Python Version |
| ----------------------- | -------------------------- | --------- | ------ | -------------- |
| 4.6.1                   | PyTorch 1.7.1              | inference | CPU    | 3.6            |
| 4.6.1                   | PyTorch 1.7.1              | inference | GPU    | 3.6            |
| 4.6.1                   | TensorFlow 2.4.1           | inference | CPU    | 3.7            |
| 4.6.1                   | TensorFlow 2.4.1           | inference | GPU    | 3.7            |
| 4.10.2                  | PyTorch 1.8.1              | inference | GPU    | 3.6            |
| 4.10.2                  | PyTorch 1.9.0              | inference | GPU    | 3.8            |
| 4.10.2                  | TensorFlow 2.4.1           | inference | GPU    | 3.7            |
| 4.10.2                  | TensorFlow 2.5.1           | inference | GPU    | 3.7            |
| 4.10.2                  | PyTorch 1.8.1              | inference | CPU    | 3.6            |
| 4.10.2                  | PyTorch 1.9.0              | inference | CPU    | 3.8            |
| 4.10.2                  | TensorFlow 2.4.1           | inference | CPU    | 3.7            |
| 4.10.2                  | TensorFlow 2.5.1           | inference | CPU    | 3.7            |
| 4.11.0                  | PyTorch 1.9.0              | inference | GPU    | 3.8            |
| 4.11.0                  | TensorFlow 2.5.1           | inference | GPU    | 3.7            |
| 4.11.0                  | PyTorch 1.9.0              | inference | CPU    | 3.8            |
| 4.11.0                  | TensorFlow 2.5.1           | inference | CPU    | 3.7            |
| 4.12.3                  | PyTorch 1.9.1              | inference | GPU    | 3.8            |
| 4.12.3                  | TensorFlow 2.5.1           | inference | GPU    | 3.7            |
| 4.12.3                  | PyTorch 1.9.1              | inference | CPU    | 3.8            |
| 4.12.3                  | TensorFlow 2.5.1           | inference | CPU    | 3.7            |

## Inference Toolkit API

The Inference Toolkit accepts inputs in the `inputs` key, and supports additional [`pipelines`](https://huggingface.co/transformers/main_classes/pipelines.html) parameters in the `parameters` key. You can provide any of the supported `kwargs` from `pipelines` as `parameters`.

Tasks supported by the Inference Toolkit API include:

- **`text-classification`**
- **`sentiment-analysis`**
- **`token-classification`**
- **`feature-extraction`**
- **`fill-mask`**
- **`summarization`**
- **`translation_xx_to_yy`**
- **`text2text-generation`**
- **`text-generation`**

See the following request examples for some of the tasks:

**`text-classification`**

```json
{
  "inputs": "This sound track was beautiful! It paints the senery in your mind so well I would recomend it
  even to people who hate vid. game music!"
}
```

**`sentiment-analysis`**

```json
{
  "inputs": "Don't waste your time.  We had two different people come to our house to give us estimates for
a deck (one of them the OWNER).  Both times, we never heard from them.  Not a call, not the estimate, nothing."
}
```

**`token-classification`**

```json
{
  "inputs": "My name is Sylvain and I work at Hugging Face in Brooklyn."
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

**`parameterized-request`**

```json
{
  "inputs": "Hugging Face, the winner of VentureBeat’s Innovation in Natural Language Process/Understanding Award for 2021, is looking to level the playing field. The team, launched by Clément Delangue and Julien Chaumond in 2016, was recognized for its work in democratizing NLP, the global market value for which is expected to hit $35.1 billion by 2026. This week, Google’s former head of Ethical AI Margaret Mitchell joined the team.",
  "paramters": {
    "repetition_penalty": 4.0,
    "length_penalty": 1.5
  }
}
```

## Inference Toolkit environment variables

The Inference Toolkit implements various additional environment variables to simplify deployment. A complete list of Hugging Face specific environment variables is shown below:

**`HF_TASK`**

`HF_TASK` defines the task for the 🤗 Transformers pipeline used . See [here](https://huggingface.co/transformers/main_classes/pipelines.html) for a complete list of tasks.

```bash
HF_TASK="question-answering"
```

**`HF_MODEL_ID`**

`HF_MODEL_ID` defines the model ID which is automatically loaded from [hf.co/models](https://huggingface.co/models) when creating a SageMaker endpoint. All of the 🤗 Hub's 10,000+ models are available through this environment variable.

```bash
HF_MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english"
```

**`HF_MODEL_REVISION`**

`HF_MODEL_REVISION` is an extension to `HF_MODEL_ID` and allows you to define or pin a model revision to make sure you always load the same model on your SageMaker endpoint.

```bash
HF_MODEL_REVISION="03b4d196c19d0a73c7e0322684e97db1ec397613"
```

**`HF_API_TOKEN`**

`HF_API_TOKEN` defines your Hugging Face authorization token. The `HF_API_TOKEN` is used as a HTTP bearer authorization for remote files like private models. You can find your token under [Settings](https://huggingface.co/settings/token) of your Hugging Face account.

```bash
HF_API_TOKEN="api_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```
