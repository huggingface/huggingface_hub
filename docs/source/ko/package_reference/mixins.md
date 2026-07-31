<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 믹스인 & 직렬화 메소드[[mixins--serialization-methods]]

## 믹스인[[mixins]]

`huggingface_hub` 라이브러리는 객체에 함수들의 업로드 및 다운로드 기능을 손쉽게 제공하기 위해서, 부모 클래스로 사용될 수 있는 다양한 믹스인을 제공합니다.
ML 프레임워크를 Hub와 통합하는 방법은 [통합 가이드](../guides/integrations)를 통해 배울 수 있습니다.

### 제네릭[[huggingface_hub.ModelHubMixin]]

[[autodoc]] ModelHubMixin
    - all
    - _save_pretrained
    - _from_pretrained

### PyTorch[[huggingface_hub.PyTorchModelHubMixin]]

[[autodoc]] PyTorchModelHubMixin

### Fastai[[huggingface_hub.from_pretrained_fastai]]

[[autodoc]] from_pretrained_fastai

[[autodoc]] push_to_hub_fastai
