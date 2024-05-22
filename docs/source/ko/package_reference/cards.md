# 리포지토리 카드[[repository-cards]]

huggingface_hub 라이브러리는 모델/데이터 세트 카드를 생성, 공유 및 업데이트하기 위한 Python 인터페이스를 제공합니다.
Hub의 모델 카드가 무엇이며 내부적으로 어떻게 작동하는지 더 깊이 있게 알아보려면 [전용 문서 페이지](https://huggingface.co/docs/hub/models-cards)를 방문하세요. 또한 이러한 유틸리티를 자신의 프로젝트에서 어떻게 사용할 수 있는지 감을 잡기 위해 [모델 카드 가이드](../how-to-model-cards)를 확인할 수 있습니다.

## 리포지토리 카드[[huggingface_hub.RepoCard]]

`RepoCard` 객체는 [`ModelCard`], [`DatasetCard`] 및 `SpaceCard`의 상위 클래스입니다.

[[autodoc]] huggingface_hub.repocard.RepoCard
    - __init__
    - all

## 카드 데이터[[huggingface_hub.CardData]]

[`CardData`] 객체는 [`ModelCardData`]와 [`DatasetCardData`]의 상위 클래스입니다.

[[autodoc]] huggingface_hub.repocard_data.CardData

## 모델 카드[[model-cards]]

### ModelCard[[huggingface_hub.ModelCard]]

[[autodoc]] ModelCard

### ModelCardData[[huggingface_hub.ModelCardData]]

[[autodoc]] ModelCardData

## 데이터 세트 카드[[cards#dataset-cards]]

ML 커뮤니티에서는 데이터 세트 카드를 데이터 카드라고도 합니다.

### DatasetCard[[huggingface_hub.DatasetCard]]

[[autodoc]] DatasetCard

### DatasetCardData[[huggingface_hub.DatasetCardData]]

[[autodoc]] DatasetCardData

## 공간 카드[[space-cards]]

### SpaceCard[[huggingface_hub.SpaceCardData]]

[[autodoc]] SpaceCard

### SpaceCardData[[huggingface_hub.SpaceCardData]]

[[autodoc]] SpaceCardData

## 유틸리티[[utilities]]

### EvalResult[[huggingface_hub.EvalResult]]

[[autodoc]] EvalResult

### model_index_to_eval_results[[huggingface_hub.repocard_data.model_index_to_eval_results]]

[[autodoc]] huggingface_hub.repocard_data.model_index_to_eval_results

### eval_results_to_model_index[[huggingface_hub.repocard_data.eval_results_to_model_index]]

[[autodoc]] huggingface_hub.repocard_data.eval_results_to_model_index

### metadata_eval_result[[huggingface_hub.metadata_eval_result]]

[[autodoc]] huggingface_hub.repocard.metadata_eval_result

### metadata_update[[huggingface_hub.metadata_update]]

[[autodoc]] huggingface_hub.repocard.metadata_update