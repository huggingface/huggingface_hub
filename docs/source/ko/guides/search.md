<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 허브에서 검색하기

이 튜토리얼에서는 `huggingface_hub`를 사용하여 허브에서 모델, 데이터 세트 및 공간을 검색하는 방법을 배웁니다.

## 레포지토리를 어떻게 나열하나요?

`huggingface_hub` 라이브러리에는 허브와 상호작용하기 위한 HTTP 클라이언트[`HfApi`]가 포함되어 있습니다.
이를 통해, 허브에 저장된 모델, 데이터셋, 그리고 스페이스들을 나열할 수 있습니다.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[`list_models`]의 출력은 허브에 저장된 모델들을 순회할 수 있는 반복자입니다.

마찬가지로, [`list_datasets`]를 사용하여 데이터셋을 나열하고 [`list_spaces`]를 사용하여 스페이스를 나열할 수 있습니다.

## 레포지토리를 어떻게 필터링하나요?

레포지토리를 나열하는 것도 유용하지만, 이제 검색을 필터링하고 싶을 수도 있습니다.
목록 도우미에는 다음과 같은 여러 속성이 있습니다.
- `filter`
- `author`
- `search`
- ...

이 매개변수 중 두 개는 직관적입니다(`author` 및 `search`). 그렇다면 `filter`는 어떤 것을 나타낼까요?
`filter`는 [`ModelFilter`] 객체(또는 [`DatasetFilter`])를 입력으로 받습니다. 이를 이용해 필터링 하고 싶은 모델을 지정하여 인스턴스를 생성할 수 있습니다.

PyTorch로 작동되고 imagenet으로 훈련된 이미지 분류에 대한 Hub의 모든 모델을 찾는 방법으로 예를 들어보겠습니다. 이 과정은 단일 [ModelFilter]를 사용하여 수행할 수 있습니다. 이때, 필터링 속성들은 '논리적 AND'로 결합되어, 지정한 모든 조건을 만족하는 모델만 선택됩니다.

```py
models = hf_api.list_models(
    filter=ModelFilter(
		task="image-classification",
		library="pytorch",
		trained_dataset="imagenet"
	)
)
```

필터링하는 과정에서 모델을 정렬하고 상위 결과만 선택할 수도 있습니다. 다음 예제는 허브에서 가장 많이 다운로드된 상위 5개 데이터셋을 가져옵니다.

```py
>>> list(list_datasets(sort="downloads", direction=-1, limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



허브에서 사용 가능한 필터에 대해 살펴보려면 웹브라우저에서 [모델](https://huggingface.co/models) 및 [데이터셋](https://huggingface.co/datasets) 페이지를 방문하여 일부 매개변수를 검색한 다음, URL에서 값들을 확인해보세요.
