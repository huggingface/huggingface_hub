---
language:
- {lang_0}  # Example: fr
- {lang_1}  # Example: en
license: {license}  # Example: apache-2.0 or any license from https://hf.co/docs/hub/model-repos#list-of-license-identifiers
library_name: {library_name}  # Example: keras or any library from https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Libraries.ts
tags:
- {tag_0}  # Example: audio
- {tag_1}  # Example: automatic-speech-recognition
- {tag_2}  # Example: speech
- {tag_3}  # Example to specify a library: allennlp
datasets:
- {dataset_0}  # Example: common_voice. Use dataset id from https://hf.co/datasets
metrics:
- {metric_0}  # Example: wer. Use metric id from https://hf.co/metrics

# Optional. Add this if you want to encode your eval results in a structured way.
model-index:
- name: {model_id}
  results:
  - task:
      type: {task_type}  # Required. Example: automatic-speech-recognition
      name: {task_name}  # Optional. Example: Speech Recognition
    dataset:
      type: {dataset_type}         # Required. Example: common_voice. Use dataset id from https://hf.co/datasets
      name: {dataset_name}         # Required. Example: Common Voice (French)
      config: {dataset_config}     # Optional. Example: fr
      split: {dataset_split}       # Optional. Example: test
      revision: {dataset_revision} # Optional. Example: 5503434ddd753f426f4b38109466949a1217c2bb
      args: 
        {arg_0: value_0}           # Optional. Example for wikipedia: language: en
        {arg_1: value_1}           # Optional. Example for wikipedia: data: 20220301
    metrics:
      - type: {metric_type}    # Required. Example: wer
        value: {metric_value}  # Required. Example: 20.90
        name: {metric_name}    # Optional. Example: Test WER
        args: {arg_0}          # Optional. Example for BLEU: max_order
---

This markdown file contains the spec for the modelcard metadata.
When present, and only then, 'model-index', 'datasets' and 'license' contents will be verified when git pushing changes to your README.md file.
Valid license identifiers can be found in [our docs](https://hf.co/docs/hub/model-repos#list-of-license-identifiers)
