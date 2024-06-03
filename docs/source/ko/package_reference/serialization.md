<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 직렬화[[serialization]]

`huggingface_hub`에는 ML 라이브러리가 모델 가중치를 표준화된 방식으로 직렬화 할 수 있도록 돕는 헬퍼를 포함하고 있습니다. 라이브러리의 이 부분은 아직 개발 중이며 향후 버전에서 개선될 예정입니다. 개선 목표는 Hub에서 가중치의 직렬화 방식을 통일하고, 라이브러리 간 코드 중복을 줄이며, Hub에서의 규약을 촉진하는 것입니다.

## 상태 사전을 샤드로 나누기[[split-state-dict-into-shards]]

현재 이 모듈은 상태 딕셔너리(예: 레이어 이름과 관련 텐서 간의 매핑)를 받아 여러 샤드로 나누고, 이 과정에서 적절한 인덱스를 생성하는 단일 헬퍼를 포함하고 있습니다. 이 헬퍼는 `torch`, `tensorflow`, `numpy` 텐서에 사용 가능하며, 다른 ML 프레임워크로 쉽게 확장될 수 있도록 설계되었습니다.

### split_numpy_state_dict_into_shards[[huggingface_hub.split_numpy_state_dict_into_shards]]

[[autodoc]] huggingface_hub.split_numpy_state_dict_into_shards

### split_tf_state_dict_into_shards[[huggingface_hub.split_tf_state_dict_into_shards]]

[[autodoc]] huggingface_hub.split_tf_state_dict_into_shards

### split_torch_state_dict_into_shards[[huggingface_hub.split_torch_state_dict_into_shards]]

[[autodoc]] huggingface_hub.split_torch_state_dict_into_shards

### split_state_dict_into_shards_factory[[huggingface_hub.split_state_dict_into_shards_factory]]

이것은 각 프레임워크별 헬퍼가 파생되는 기본 틀입니다. 실제로는 아직 지원되지 않는 프레임워크에 맞게 조정할 필요가 있는 경우가 아니면 이 틀을 직접 사용할 것으로 예상되지 않습니다. 그런 경우가 있다면, `huggingface_hub` 리포지토리에 [새로운 이슈를 개설](https://github.com/huggingface/huggingface_hub/issues/new) 하여 알려주세요.

[[autodoc]] huggingface_hub.split_state_dict_into_shards_factory

## 도우미

### get_torch_storage_id[[huggingface_hub.get_torch_storage_id]]

[[autodoc]] huggingface_hub.get_torch_storage_id