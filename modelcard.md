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
- {dataset_0}  # Example: common_voice
metrics:
- {metric_0}  # Example: wer

model-index:  
- name: {model_id}
  results:
  - task: 
      name: {task_name}  # Example: Speech Recognition
      type: {task_type}  # Example: automatic-speech-recognition
    dataset:
      name: {dataset_name}  # Example: Common Voice zh-CN
      type: {dataset_type}  # Example: common_voice
      args: {arg_0}  # Example: zh-CN
    metrics:
      - name: {metric_name}  # Example: Test WER
        type: {metric_type}  # Example: wer
        value: {metric_value}  # Example: 20.90
        args: {arg_0}  # Example for BLEU: max_order
---

This markdown file contains the spec for the modelcard metadata regarding evaluation parameters. 