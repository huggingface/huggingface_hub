<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Mixins & méthodes de sérialisation

## Mixins

La librairie `huggingface_hub` offre une liste de mixins qui peuvent être utilisés en tant que classes parentes pour vos
objets, afin d'avoir des fonctions upload et de téléchargements simples. Consultez notre [guide d'intégration](../guides/integrations)
pour apprendre à intégrer n'importe quel framework ML avec le Hub.

### Generic

[[autodoc]] ModelHubMixin
    - all
    - _save_pretrained
    - _from_pretrained

### PyTorch

[[autodoc]] PyTorchModelHubMixin

### Keras

[[autodoc]] KerasModelHubMixin

[[autodoc]] from_pretrained_keras

[[autodoc]] push_to_hub_keras

[[autodoc]] save_pretrained_keras

### Fastai

[[autodoc]] from_pretrained_fastai

[[autodoc]] push_to_hub_fastai



