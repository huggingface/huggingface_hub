<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 허브 검색

이 튜토리얼에서는 `huggingface_hub`를 사용하여 허브에서 모델, 데이터 세트 및 공간을 검색하는 방법을 배웁니다.

## 저장소를 나열하는 방법은 무엇입니까?

`huggingface_hub` 라이브러리에는 허브와 상호작용하기 위한 HTTP 클라이언트[`HfApi`]가 포함되어 있습니다.
무엇보다도 허브에 저장된 모델, 데이터세트 및 공간을 나열할 수 있습니다.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[`list_models`]의 출력은 허브에 저장된 모델에 대한 반복자입니다.

마찬가지로 [`list_datasets`]를 사용하여 데이터세트를 나열하고 [`list_spaces`]를 사용하여 Space를 나열할 수 있습니다.

## 저장소를 필터링하는 방법은 무엇입니까?

리포지토리를 나열하는 것은 훌륭하지만 이제 검색을 필터링하고 싶을 수도 있습니다.
목록 도우미에는 다음과 같은 여러 속성이 있습니다.
- `필터`
- `저자`
- `검색`
- ...

이 매개변수 중 두 개는 직관적입니다(`author` 및 `search`). 하지만 `filter`는 어떻습니까?
`filter`는 [`ModelFilter`] 객체(또는 [`DatasetFilter`])를 입력으로 사용합니다. 인스턴스화할 수 있습니다.
필터링하려는 모델을 지정하여 이를 수행합니다.

이미지 분류를 수행하는 허브의 모든 모델을 가져오는 예를 살펴보겠습니다.
imagenet 데이터 세트에 대해 교육을 받았으며 PyTorch와 함께 실행됩니다. 그거 하나만으로 가능해요
[`ModelFilter`]. 속성은 "논리적 AND"로 결합됩니다.

```py
models = hf_api.list_models(
    filter=ModelFilter(
		task="image-classification",
		library="pytorch",
		trained_dataset="imagenet"
	)
)
```

필터링하는 동안 모델을 정렬하고 상위 결과만 가져올 수도 있습니다. 예를 들어,
다음 예에서는 허브에서 가장 많이 다운로드된 상위 5개 데이터 세트를 가져옵니다.

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



허브에서 사용 가능한 필터를 살펴보려면 브라우저에서 [models](https://huggingface.co/models) 및 [datasets](https://huggingface.co/datasets) 페이지를 방문하여 일부 매개변수를 검색하고 URL의 값들을 살펴보세요.