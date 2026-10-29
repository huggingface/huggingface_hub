<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 모델 카드 생성 및 공유[[create-and-share-model-cards]]

`huggingface_hub` 라이브러리는 모델 카드를 생성, 공유, 업데이트할 수 있는 파이썬 인터페이스를 제공합니다. Hub의 모델 카드가 무엇인지, 그리고 실제로 어떻게 작동하는지에 대한 자세한 내용을 확인하려면 [전용 설명 페이지](https://huggingface.co/docs/hub/models-cards)를 방문하세요.

> [!TIP]
> [신규 (베타)! 우리의 실험적인 모델 카드 크리에이터 앱을 사용해 보세요](https://huggingface.co/spaces/huggingface/Model_Cards_Writing_Tool)

## Hub에서 모델 카드 불러오기[[load-a-model-card-from-the-hub]]

Hub에서 기존 카드를 불러오려면 [`ModelCard.load`] 기능을 사용하면 됩니다. 이 문서에서는 [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans)에서 카드를 불러오겠습니다.


```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

이 카드에는 접근하거나 활용할 수 있는 몇 가지 유용한 속성이 있습니다:

  - `card.data`: 모델 카드의 메타데이터와 함께 [`ModelCardData`] 인스턴스를 반환합니다. 이 인스턴스에 `.to_dict()`를 호출하여 표현을 사전으로 가져옵니다.
  - `card.text`: *메타데이터 헤더를 제외*한 카드의 텍스트를 반환합니다.
  - `card.content`: *메타데이터 헤더를 포함*한 카드의 텍스트 콘텐츠를 반환합니다.

## 모델 카드 만들기[[create-model-cards]]

### 텍스트에서 생성[[from-text]]

텍스트로 모델 카드의 초기 내용을 설정하려면, 카드의 텍스트 내용을 초기화 시 `ModelCard`에 전달하면 됩니다.

```python
content = """
---
language: en
license: mit
---

# 내 모델 카드
"""

card = ModelCard(content)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True
```
 
이 작업을 수행하는 또 다른 방법은 f-strings를 사용하는 것입니다. 다음 예에서 우리는:

- 모델 카드에 YAML 블록을 삽입할 수 있도록 [`ModelCardData.to_yaml`]을 사용해서 우리가 정의한 메타데이터를 YAML로 변환합니다.
- Python f-strings를 통해 템플릿 변수를 사용할 방법을 보여줍니다.

```python
card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
content = f"""
---
{ card_data.to_yaml() }
---

# 내 모델 카드

이 모델은 [@{example_template_var}](https://github.com/ {example_template_var})에 의해 생성되었습니다
"""

card = ModelCard(content)
print(card)
```

위 예시는 다음과 같은 모습의 카드를 남깁니다:

```
---
language: en
license: mit
library: timm
---

# 내 모델 카드

This model created by [@nateraw](https://github.com/nateraw)
```

### Jinja 템플릿으로부터[[from-a-jinja-template]]

`Jinja2`가 설치되어 있으면, jinja 템플릿 파일에서 모델 카드를 만들 수 있습니다. 기본적인 예를 살펴보겠습니다:

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# jinja 템플릿 정의
template_text = """
---
{{ card_data }}
---

# MyCoolModel 모델용 모델 카드

이 모델은 이런 저런 것들을 합니다.

이 모델은 [[@{{ author }}](https://hf.co/{{author}})에 의해 생성되었습니다.
""".strip() 

# 템플릿을 파일에 쓰기
Path('custom_template.md').write_text(template_text)

# 카드 메타데이터 정의
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# 템플릿에서 카드를 만들고 원하는 Jinja 템플릿 변수를 전달합니다.
# 우리의 경우에는 작성자를 전달하겠습니다.
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

결과 카드의 마크다운은 다음과 같습니다:

```
---
language: en
license: mit
library_name: keras
---

# MyCoolModel 모델용 모델 카드

이 모델은 이런 저런 것들을 합니다.

이 모델은 [@nateraw](https://hf.co/nateraw)에 의해 생성되었습니다.
```

카드 데이터를 업데이트하면 카드 자체에 반영됩니다.

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

이제 보시다시피 메타데이터 헤더가 업데이트되었습니다:

```
---
language: fr
license: apache-2.0
library_name: timm
---

# MyCoolModel 모델용 모델 카드

이 모델은 이런 저런 것들을 합니다.

이 모델은 [@nateraw](https://hf.co/nateraw)에 의해 생성되었습니다.
```

카드 데이터를 업데이트할 때 [`ModelCard.validate`]를 불러와 Hub에 대해 카드가 여전히 유효한지 확인할 수 있습니다. 이렇게 하면 Hugging Face Hub에 설정된 모든 유효성 검사 규칙을 통과할 수 있습니다.

### 기본 템플릿으로부터[[from-the-default-template]]

자체 템플릿을 사용하는 대신에, 많은 섹션으로 구성된 기능이 풍부한 [기본 템플릿](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)을 사용할 수도 있습니다. 내부적으론 [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) 를 사용하여 템플릿 파일을 작성합니다.

> [!TIP]
> `from_template`를 사용하려면 jinja2를 설치해야 합니다. `pip install Jinja2`를 사용하면 됩니다.

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
card.save('my_model_card_2.md')
print(card)
```

## 모델 카드 공유하기[[share-model-cards]]

Hugging Face Hub로 인증받은 경우(`hf auth login` 또는 [`login`] 사용) 간단히 [`ModelCard.push_to_hub`]를 호출하여 카드를 Hub에 푸시할 수 있습니다. 이를 수행하는 방법을 살펴보겠습니다.

먼저 인증된 사용자의 네임스페이스 아래에 'hf-hub-modelcards-pr-test'라는 새로운 레포지토리를 만듭니다:

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

그런 다음 기본 템플릿에서 카드를 만듭니다(위 섹션에서 정의한 것과 동일):

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
```

마지막으로 이를 Hub로 푸시하겠습니다.

```python
card.push_to_hub(repo_id)
```

결과 카드는 [여기](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md)에서 확인할 수 있습니다.

PR로 카드를 푸시하고 싶다면 `push_to_hub`를 호출할 때 `create_pr=True`라고 지정하면 됩니다.

```python
card.push_to_hub(repo_id, create_pr=True)
```

이 명령으로 생성된 결과 PR은 [여기](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3)에서 볼 수 있습니다.

## 메타데이터 업데이트[[update-metadata]]

이 섹션에서는 레포 카드에 있는 메타데이터와 업데이트 방법을 확인합니다.

`메타데이터`는 모델, 데이터 세트, Spaces에 대한 높은 수준의 정보를 제공하는 해시맵(또는 키 값) 컨텍스트를 말합니다. 모델의 `pipeline type`, `model_id` 또는 `model_desc` 설명 등의 정보가 포함될 수 있습니다. 자세한 내용은 [모델 카드](https://huggingface.co/docs/hub/model-cards#model-card-metadata), [데이터 세트 카드](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) 및 [�Spaces 설정](https://huggingface.co/docs/hub/spaces-settings#spaces-settings) 을 참조하세요. 이제 메타데이터를 업데이트하는 방법에 대한 몇 가지 예를 살펴보겠습니다.


첫 번째 예부터 살펴보겠습니다:

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

두 줄의 코드를 사용하면 메타데이터를 업데이트하여 새로운 `파이프라인_태그`를 설정할 수 있습니다.

기본적으로 카드에 이미 존재하는 키는 업데이트할 수 없습니다. 그렇게 하려면 `overwrite=True`를 명시적으로 전달해야 합니다.

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

쓰기 권한이 없는 저장소에 일부 변경 사항을 제안하려는 경우가 종종 있습니다. 소유자가 귀하의 제안을 검토하고 병합할 수 있도록 해당 저장소에 PR을 생성하면 됩니다.

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## 평가 결과 포함하기[[include-evaluation-results]]

메타데이터 `모델-인덱스`에 평가 결과를 포함하려면 관련 평가 결과와 함께 [EvalResult] 또는 `EvalResult` 목록을 전달하면 됩니다. 내부적으론 `card.data.to _dict()`를 호출하면 `모델-인덱스`가 생성됩니다. 자세한 내용은 [Hub 문서의 이 섹션](https://huggingface.co/docs/hub/models-cards#evaluation-results)을 참조하십시오.

> [!TIP]
> 이 기능을 사용하려면 [ModelCardData]에 `model_name` 속성을 포함해야 합니다.

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = EvalResult(
        task_type='image-classification',
        dataset_type='beans',
        dataset_name='Beans',
        metric_type='accuracy',
        metric_value=0.7
    )
)

card = ModelCard.from_template(card_data)
print(card.data)
```

결과 `card.data`는 다음과 같이 보여야 합니다:

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
```

`EvalResult`: 공유하고 싶은 평가 결과가 둘 이상 있는 경우 `EvalResults` 목록을 전달하기만 하면 됩니다:

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = [
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.7
        ),
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='f1',
            metric_value=0.65
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data
```
그러면 다음 `card.data`가 남게 됩니다:

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65
```
