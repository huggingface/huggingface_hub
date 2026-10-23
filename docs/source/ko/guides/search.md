<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hub에서 검색하기[[search-the-hub]]

이 튜토리얼에서는 `huggingface_hub`를 사용하여 Hub에서 모델, 데이터 세트 및 Spaces를 검색하는 방법을 배웁니다.

## 리포지토리를 어떻게 나열하나요?[[how-to-list-repositories-]]

`huggingface_hub` 라이브러리에는 Hub와 상호작용하기 위한 HTTP 클라이언트[`HfApi`]가 포함되어 있습니다.
이를 통해, Hub에 저장된 모델, 데이터셋, 그리고 Spaces를 나열할 수 있습니다.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[`list_models`]의 출력은 Hub에 저장되어 있는 모델들을 나열한 결과입니다.

마찬가지로, [`list_datasets`]를 사용하여 데이터 세트를 나열하고 [`list_spaces`]를 사용하여 Spaces를 나열할 수 있습니다.

## 리포지토리를 어떻게 필터링하나요?[[how-to-filter-repositories-]]

리포지토리를 나열하는 것도 유용하지만, 검색을 필터링하고 싶을 수도 있습니다.
리스트에는 다음과 같은 여러 속성이 있습니다.
- `filter`
- `author`
- `search`
- ...

`filter`는 태그 문자열이나 태그 목록을 받습니다. 작업과 훈련 데이터 세트는 각각 `pipeline_tag`와 `trained_dataset`으로 지정합니다.

다음 예시는 imagenet 데이터 세트로 훈련된 PyTorch 이미지 분류 모델을 찾습니다. 조건은 논리적 AND로 결합됩니다.

```py
models = api.list_models(
    pipeline_tag="image-classification",
    filter="pytorch",
    trained_dataset="imagenet",
)
```

필터링하는 과정에서 모델을 정렬하고 상위 결과만 선택할 수도 있습니다. 다음 예제는 Hub에서 가장 많이 다운로드된 상위 5개 데이터 세트를 가져옵니다.

```py
>>> list(api.list_datasets(sort="downloads", limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



Hub에서 사용 가능한 필터에 대해 살펴보려면 웹브라우저에서 [모델](https://huggingface.co/models) 및 [데이터 세트](https://huggingface.co/datasets) 페이지를 방문하여 일부 매개변수를 검색한 다음, URL에서 값들을 확인해보세요.
