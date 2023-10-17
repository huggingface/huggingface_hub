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
    metrics:
    - type: acc
      value: 0.9
---

# Invalid Model Index

In this example, the model index does not define a dataset field. In this case, we'll still initialize CardData, but will leave model-index/eval_results out of it.
