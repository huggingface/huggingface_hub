<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 서버에서 추론 진행하기[[run-inference-on-servers]]

추론은 훈련된 모델을 사용하여 새 데이터에 대한 예측을 수행하는 과정입니다. 이 과정은 계산이 많이 필요할 수 있으므로, 전용 서버에서 실행하는 것이 좋은 방안이 될 수 있습니다. `huggingface_hub` 라이브러리는 호스팅된 모델에 대한 추론을 실행하는 서비스를 호출하는 간편한 방법을 제공합니다. 다음과 같은 여러 서비스에 연결할 수 있습니다:
- [추론 API](https://huggingface.co/docs/api-inference/index): Hugging Face의 인프라에서 가속화된 추론을 실행할 수 있는 서비스로 무료로 제공됩니다. 이 서비스는 추론을 시작하고 다양한 모델을 테스트하며 AI 제품의 프로토타입을 만드는 빠른 방법입니다.
- [추론 엔드포인트](https://huggingface.co/docs/inference-endpoints/index): 모델을 제품 환경에 쉽게 배포할 수 있는 제품입니다. 사용자가 선택한 클라우드 환경에서 완전 관리되는 전용 인프라에서 Hugging Face를 통해 추론이 실행됩니다.


> [!TIP]
> [`InferenceClient`]는 API에 HTTP 호출을 수행하는 Python 클라이언트입니다. HTTP 호출을 원하는 툴을 이용하여 직접 사용하려면 (curl, postman 등) [추론 API](https://huggingface.co/docs/api-inference/index) 또는 [추론 엔드포인트](https://huggingface.co/docs/inference-endpoints/index) 문서 페이지를 참조하세요.
>
> 웹 개발을 위해 [JS 클라이언트](https://huggingface.co/docs/huggingface.js/inference/README)가 출시되었습니다. 게임 개발에 관심이 있다면 [C# 프로젝트](https://github.com/huggingface/unity-api)를 살펴보세요.

## 시작하기[[getting-started]]

text-to-image 작업을 시작해보겠습니다.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> image = client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")
```

우리는 기본 매개변수로 [`InferenceClient`]를 초기화했습니다. 수행하고자 하는 [작업](#supported-tasks)만 알면 됩니다. 기본적으로 클라이언트는 추론 API에 연결하고 작업을 완료할 모델을 선택합니다. 예제에서는 텍스트 프롬프트에서 이미지를 생성했습니다. 반환된 값은 파일로 저장할 수 있는 `PIL.Image` 객체입니다.

> [!WARNING]
> API는 간단하게 설계되었습니다. 모든 매개변수와 옵션이 사용 가능하거나 설명되어 있는 것은 아닙니다. 각 작업에서 사용 가능한 모든 매개변수에 대해 자세히 알아보려면 [이 페이지](https://huggingface.co/docs/api-inference/detailed_parameters)를 확인하세요.

### 특정 모델 사용하기[[using-a-specific-model]]

특정 모델을 사용하고 싶다면 어떻게 해야 할까요? 매개변수로 직접 지정하거나 인스턴스 수준에서 직접 지정할 수 있습니다:

```python
>>> from huggingface_hub import InferenceClient
# 특정 모델을 위한 클라이언트를 초기화합니다.
>>> client = InferenceClient(model="prompthero/openjourney-v4")
>>> client.text_to_image(...)
# 또는 일반적인 클라이언트를 사용하되 모델을 인수로 전달하세요.
>>> client = InferenceClient()
>>> client.text_to_image(..., model="prompthero/openjourney-v4")
```

> [!TIP]
> Hugging Face Hub에는 20만 개가 넘는 모델이 있습니다! [`InferenceClient`]의 각 작업에는 추천되는 모델이 포함되어 있습니다. HF의 추천은 사전 고지 없이 시간이 지남에 따라 변경될 수 있음을 유의하십시오. 따라서 모델을 결정한 후에는 명시적으로 모델을 설정하는 것이 좋습니다. 또한 대부분의 경우 자신의 필요에 맞는 모델을 직접 찾고자 할 것입니다. 허브의 [모델](https://huggingface.co/models) 페이지를 방문하여 찾아보세요.

### 특정 URL 사용하기[[using-a-specific-url]]

위에서 본 예제들은 서버리스 추론 API를 사용합니다. 이는 빠르게 프로토타입을 정하고 테스트할 때 매우 유용합니다. 모델을 프로덕션 환경에 배포할 준비가 되면 전용 인프라를 사용해야 합니다. 그것이 [추론 엔드포인트](https://huggingface.co/docs/inference-endpoints/index)가 필요한 이유입니다. 이를 사용하면 모든 모델을 배포하고 개인 API로 노출시킬 수 있습니다. 한 번 배포되면 이전과 완전히 동일한 코드를 사용하여 연결할 수 있는 URL을 얻게 됩니다. `model` 매개변수만 변경하면 됩니다:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# 또는
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

### 인증[[authentication]]

[`InferenceClient`]로 수행된 호출은 [사용자 액세스 토큰](https://huggingface.co/docs/hub/security-tokens)을 사용하여 인증할 수 있습니다. 기본적으로 로그인한 경우 기기에 저장된 토큰을 사용합니다 ([인증 방법](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)을 확인하세요). 로그인하지 않은 경우 인스턴스 매개변수로 토큰을 전달할 수 있습니다.

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(token="hf_***")
```

> [!TIP]
> 추론 API를 사용할 때 인증은 필수가 아닙니다. 그러나 인증된 사용자는 서비스를 이용할 수 있는 더 높은 무료 티어를 받습니다. 토큰은 개인 모델이나 개인 엔드포인트에서 추론을 실행하려면 필수입니다.

## 지원되는 작업[[supported-tasks]]

[`InferenceClient`]의 목표는 Hugging Face 모델에서 추론을 실행하기 위한 가장 쉬운 인터페이스를 제공하는 것입니다. 이는 가장 일반적인 작업들을 지원하는 간단한 API를 가지고 있습니다. 현재 지원되는 작업 목록은 다음과 같습니다:

| 도메인      | 작업                                                                              | 지원 여부 | 문서                                                |
| ----------- | --------------------------------------------------------------------------------- | --------- | --------------------------------------------------- |
| 오디오      | [오디오 분류](https://huggingface.co/tasks/audio-classification)                  | ✅         | [`~InferenceClient.audio_classification`]           |
| 오디오      | [오디오 투 오디오](https://huggingface.co/tasks/audio-to-audio)                   | ✅         | [`~InferenceClient.audio_to_audio`]                 |
|             | [자동 음성 인식](https://huggingface.co/tasks/automatic-speech-recognition)       | ✅         | [`~InferenceClient.automatic_speech_recognition`]   |
|             | [텍스트 투 스피치](https://huggingface.co/tasks/text-to-speech)                   | ✅         | [`~InferenceClient.text_to_speech`]                 |
| 컴퓨터 비전 | [이미지 분류](https://huggingface.co/tasks/image-classification)                  | ✅         | [`~InferenceClient.image_classification`]           |
|             | [이미지 분할](https://huggingface.co/tasks/image-segmentation)                    | ✅         | [`~InferenceClient.image_segmentation`]             |
|             | [이미지 투 이미지](https://huggingface.co/tasks/image-to-image)                   | ✅         | [`~InferenceClient.image_to_image`]                 |
|             | [이미지 투 텍스트](https://huggingface.co/tasks/image-to-text)                    | ✅         | [`~InferenceClient.image_to_text`]                  |
|             | [객체 탐지](https://huggingface.co/tasks/object-detection)                        | ✅         | [`~InferenceClient.object_detection`]               |
|             | [텍스트 투 이미지](https://huggingface.co/tasks/text-to-image)                    | ✅         | [`~InferenceClient.text_to_image`]                  |
|             | [제로샷 이미지 분류](https://huggingface.co/tasks/zero-shot-image-classification) | ✅         | [`~InferenceClient.zero_shot_image_classification`] |
| 멀티모달    | [문서 질의 응답](https://huggingface.co/tasks/document-question-answering)        | ✅         | [`~InferenceClient.document_question_answering`]    |
|             | [시각적 질의 응답](https://huggingface.co/tasks/visual-question-answering)        | ✅         | [`~InferenceClient.visual_question_answering`]      |
| 자연어 처리 | [대화형](https://huggingface.co/tasks/conversational)                             | ✅         | [`~InferenceClient.conversational`]                 |
|             | [특성 추출](https://huggingface.co/tasks/feature-extraction)                      | ✅         | [`~InferenceClient.feature_extraction`]             |
|             | [마스크 채우기](https://huggingface.co/tasks/fill-mask)                           | ✅         | [`~InferenceClient.fill_mask`]                      |
|             | [질의 응답](https://huggingface.co/tasks/question-answering)                      | ✅         | [`~InferenceClient.question_answering`]             |
|             | [문장 유사도](https://huggingface.co/tasks/sentence-similarity)                   | ✅         | [`~InferenceClient.sentence_similarity`]            |
|             | [요약](https://huggingface.co/tasks/summarization)                                | ✅         | [`~InferenceClient.summarization`]                  |
|             | [테이블 질의 응답](https://huggingface.co/tasks/table-question-answering)         | ✅         | [`~InferenceClient.table_question_answering`]       |
|             | [텍스트 분류](https://huggingface.co/tasks/text-classification)                   | ✅         | [`~InferenceClient.text_classification`]            |
|             | [텍스트 생성](https://huggingface.co/tasks/text-generation)                       | ✅         | [`~InferenceClient.text_generation`]                |
|             | [토큰 분류](https://huggingface.co/tasks/token-classification)                    | ✅         | [`~InferenceClient.token_classification`]           |
|             | [번역](https://huggingface.co/tasks/translation)                                  | ✅         | [`~InferenceClient.translation`]                    |
|             | [제로샷 분류](https://huggingface.co/tasks/zero-shot-classification)              | ✅         | [`~InferenceClient.zero_shot_classification`]       |
| 타블로      | [타블로 작업 분류](https://huggingface.co/tasks/tabular-classification)           | ✅         | [`~InferenceClient.tabular_classification`]         |
|             | [타블로 회귀](https://huggingface.co/tasks/tabular-regression)                    | ✅         | [`~InferenceClient.tabular_regression`]             |

> [!TIP]
> 각 작업에 대해 더 자세히 알고 싶거나 사용 방법 및 각 작업에 대한 가장 인기 있는 모델을 알아보려면 [Tasks](https://huggingface.co/tasks) 페이지를 확인하세요.



## 비동기 클라이언트[[async-client]]

`asyncio`와 `aiohttp`를 기반으로 한 클라이언트의 비동기 버전도 제공됩니다. `aiohttp`를 직접 설치하거나 `[inference]` 추가 옵션을 사용할 수 있습니다:

```sh
pip install --upgrade huggingface_hub[inference]
# 또는
# pip install aiohttp
```

설치 후 모든 비동기 API 엔드포인트는 [`AsyncInferenceClient`]를 통해 사용할 수 있습니다. 초기화 및 API는 동기 전용 버전과 완전히 동일합니다.

```py
# 코드는 비동기 asyncio 라이브러리 동시성 컨텍스트에서 실행되어야 합니다.
# $ python -m asyncio
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> async for token in await client.text_generation("The Huggingface Hub is", stream=True):
...     print(token, end="")
 a platform for sharing and discussing ML-related content.
```

`asyncio` 모듈에 대한 자세한 정보는 [공식 문서](https://docs.python.org/3/library/asyncio.html)를 참조하세요.

## 고급 팁[[advanced-tips]]

위 섹션에서는 [`InferenceClient`]의 주요 측면을 살펴보았습니다. 이제 몇 가지 고급 팁에 대해 자세히 알아보겠습니다.

### 타임아웃[[timeout]]

추론을 수행할 때 타임아웃이 발생하는 주요 원인은 두 가지입니다:
- 추론 프로세스가 완료되는 데 오랜 시간이 걸리는 경우
- 모델이 사용 불가능한 경우, 예를 들어 Inference API를 처음으로 가져오는 경우

[`InferenceClient`]에는 이 두 가지를 처리하기 위한 전역 `timeout` 매개변수가 있습니다. 기본값은 `None`으로 설정되어 있으며, 클라이언트가 추론이 완료될 때까지 무기한으로 기다리게 합니다. 워크플로우에서 더 많은 제어를 원하는 경우 초 단위의 특정한 값으로 설정할 수 있습니다. 타임아웃 딜레이가 만료되면 [`InferenceTimeoutError`]가 발생합니다. 이를 코드에서 처리할 수 있습니다:

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### 이진 입력[[binary-inputs]]

일부 작업에는 이미지 또는 오디오 파일을 처리할 때와 같이 이진 입력이 필요한 경우가 있습니다. 이 경우 [`InferenceClient`]는 최대한 다양한 유형을 융통성 있게 허용합니다:
- 원시 `bytes`
- 이진으로 열린 파일과 유사한 객체 (`with open("audio.flac", "rb") as f: ...`)
- 로컬 파일을 가리키는 경로 (`str` 또는 `Path`)
- 원격 파일을 가리키는 URL (`str`) (예: `https://...`). 이 경우 파일은 Inference API로 전송되기 전에 로컬로 다운로드됩니다.

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```
