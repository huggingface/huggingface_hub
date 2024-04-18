<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 직렬화[[serialization]]

`huggingface_hub`에는 ML 라이브러리가 모델 가중치를 표준화된 방식으로 직렬화할 수 있도록 돕는 헬퍼가 포함되어 있습니다. 이 라이브러리의 이 부분은 아직 개발 중이며 향후 릴리스에서 개선될 것입니다. 목표는 라이브러리 간 코드 중복을 제거하고 Hub의 관례를 촉진하기 위해 가중치가 어떻게 직렬화되는지를 조화시키는 것입니다.

## 상태 사전을 분할 청크로 나누기[[split-state-dict-into-shards]]

현재 이 모듈에는 상태 사전(예: 레이어 이름과 관련 텐서 간의 매핑)을 여러 청크로 분할하고 적절한 색인을 만드는 단일 헬퍼가 포함되어 있습니다. 이 헬퍼는 `torch`, `tensorflow` 및 `numpy` 텐서에 사용할 수 있도록 설계되었으며 다른 ML 프레임워크로도 쉽게 확장할 수 있습니다.

### split_numpy_state_dict_into_shards[[huggingface_hub.split_numpy_state_dict_into_shards]]

[[autodoc]] huggingface_hub.split_numpy_state_dict_into_shards

### split_tf_state_dict_into_shards[[huggingface_hub.split_tf_state_dict_into_shards]]

[[autodoc]] huggingface_hub.split_tf_state_dict_into_shards

### split_torch_state_dict_into_shards[[huggingface_hub.split_torch_state_dict_into_shards]]

[[autodoc]] huggingface_hub.split_torch_state_dict_into_shards

### split_state_dict_into_shards_factory[[huggingface_hub.split_state_dict_into_shards_factory]]

이것은 각 프레임워크별 헬퍼가 파생된 기본 팩토리입니다. 실제로 아직 지원되지 않는 프레임워크에 적응해야 하는 경우가 아니면 이 팩토리를 직접 사용할 것으로 예상되지 않습니다. 그런 경우 `huggingface_hub` 리포지토리에 [새 이슈를 여는](https://github.com/huggingface/huggingface_hub/issues/new) 것이 좋습니다.

[[autodoc]] huggingface_hub.split_state_dict_into_shards_factory