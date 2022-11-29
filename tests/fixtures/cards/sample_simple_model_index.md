---
language: en
license: mit
library_name: timm
tags:
- pytorch
- image-classification
datasets:
- beans
metrics:
- accuracy
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      type: beans
      name: Beans
    metrics:
    - type: accuracy
      value: 0.9
  - task:
      type: image-classification
    dataset:
      type: beans
      name: Beans
      config: default
      split: test
      revision: 5503434ddd753f426f4b38109466949a1217c2bb
      args:
        date: 20220120
    metrics:
    - type: f1
      value: 0.66
---

# my-cool-model

## Model description

This is a test model card with multiple evaluations across different (dataset, metric) configurations.
