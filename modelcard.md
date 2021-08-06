---
language:
- {lang_0}  # Example: fr
- {lang_1}  # Example: en
license: {license}  # Example: apache-2.0
tags:
- {tag_0}  # Example: audio
- {tag_1}  # Example: automatic-speech-recognition
- {tag_2}  # Example: speech
- {tag_3}  # Example to specify a library: allennlp
datasets:
- {dataset_0}  # Example: common_voice. Use dataset id from https://hf.co/datasets
metrics:
- {metric_0}  # Example: wer. Use metric id from https://hf.co/metrics

model-index:
- name: {model_id}
  results:
  - task: 
      type: {task_type}  # Required. Example: automatic-speech-recognition
      name: {task_name}  # Optional. Example: Speech Recognition
    dataset:
      type: {dataset_type}  # Required. Example: common_voice. Use dataset id from https://hf.co/datasets
      name: {dataset_name}  # Optional. Example: Common Voice zh-CN
      args: {arg_0}         # Optional. Example: zh-CN
    metrics:
      - type: {metric_type}    # Required. Example: wer
        value: {metric_value}  # Required. Example: 20.90
        name: {metric_name}    # Optional. Example: Test WER
        args: {arg_0}          # Optional. Example for BLEU: max_order
---

This markdown file contains the spec for the modelcard metadata regarding evaluation parameters.