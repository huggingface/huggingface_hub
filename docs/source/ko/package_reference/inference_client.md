<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 추론[[inference]]

추론은 학습된 모델을 사용하여 새로운 데이터를 예측하는 과정입니다. 이 과정은 계산량이 많을 수 있기 때문에, 전용 서버에서 실행하는 것이 흥미로운 옵션이 될 수 있습니다. `huggingface_hub` 라이브러리는 호스팅된 모델에 대한 추론을 실행하는 간단한 방법을 제공합니다. 연결할 수 있는 서비스는 여러가지가 있습니다:

- [추론 API](https://huggingface.co/docs/api-inference/index): Hugging Face의 인프라에서 가속화된 추론을 무료로 실행할 수 있는 서비스입니다. 이 서비스는 시작하기 위한 빠른 방법이며, 다양한 모델을 테스트하고 AI 제품을 프로토타입화하는 데에도 유용합니다.
- [추론 엔드포인트](https://huggingface.co/inference-endpoints): 모델을 쉽게 운영 환경으로 배포할 수 있는 제품입니다. 추론은 여러분이 선택한 클라우드 제공업체의 전용 및 완전히 관리되는 인프라에서 Hugging Face에 의해 실행됩니다.

이러한 서비스는 [`InferenceClient`] 객체를 사용하여 호출할 수 있습니다. 자세한 사용 방법에 대해서는 [이 가이드](../guides/inference)를 참조해주세요.

## 추론 클라이언트[[huggingface_hub.InferenceClient]]

[[autodoc]] InferenceClient

## 비동기 추론 클라이언트[[huggingface_hub.AsyncInferenceClient]]

비동기 버전의 클라이언트도 제공되며, 이는 `asyncio`와 `aiohttp`를 기반으로 작동합니다. 
이를 사용하려면 `aiohttp`를 직접 설치하거나 `[inference]` 추가 기능을 사용할 수 있습니다:

```sh
pip install --upgrade huggingface_hub[inference]
# 또는
# pip install aiohttp
```

[[autodoc]] AsyncInferenceClient

## 추론 시간 초과 오류[[huggingface_hub.InferenceTimeoutError]]

[[autodoc]] InferenceTimeoutError

## 반환 유형[[return-types]]

대부분의 작업에 대해, 반환 값은 내장된 유형(string, list, image...)을 갖습니다. 보다 복잡한 유형을 위한 목록은 다음과 같습니다.
