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
- acc
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      type: beans
      name: Beans
    metrics:
    - type: acc
      value: 0.9
---

# my-cool-model

## Model description

You can embed local or remote images using `![](...)`
