<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Mixins & serialization methods[[mixins--serialization-methods]]

## Mixins[[mixins]]

`huggingface_hub` 라이브러리는 객체에 함수들의 업로드 및 다운로드 기능을 손쉽게 제공하기 위해서, 부모 클래스로 사용될 수 있는 다양한 믹스인을 제공합니다.
ML 프레임워크를 Hub와 통합하는 방법은 [통합 가이드](../guides/integrations)를 통해 배울 수 있습니다.

### Generic[[huggingface_hub.ModelHubMixin]]

[[autodoc]] ModelHubMixin
    - all
    - _save_pretrained
    - _from_pretrained

### PyTorch[[huggingface_hub.PyTorchModelHubMixin]]

[[autodoc]] PyTorchModelHubMixin

### Keras[[huggingface_hub.KerasModelHubMixin]]

[[autodoc]] KerasModelHubMixin

[[autodoc]] from_pretrained_keras

[[autodoc]] push_to_hub_keras

[[autodoc]] save_pretrained_keras

### Fastai[[huggingface_hub.from_pretrained_fastai]]

[[autodoc]] from_pretrained_fastai

[[autodoc]] push_to_hub_fastai
