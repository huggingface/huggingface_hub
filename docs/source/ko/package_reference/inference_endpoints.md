# 추론 엔드포인트 [[inference-endpoints]]

Hugging Face가 관리하는 추론 엔드포인트는 우리가 모델을 쉽고 안전하게 배포할 수 있게 해주는 도구입니다. 이러한 추론 엔드포인트는 [Hub](https://huggingface.co/models)에 있는 모델을 기반으로 설계되었습니다. 이 문서는 `huggingface_hub`와 추론 엔드포인트 통합에 관한 참조 페이지이며, 더욱 자세한 정보는 [공식 문서](https://huggingface.co/docs/inference-endpoints/index)를 통해 확인할 수 있습니다.

> [!TIP]
> 'huggingface_hub'를 사용하여 추론 엔드포인트를 프로그래밍 방식으로 관리하는 방법을 알고 싶다면, [관련 가이드](../guides/inference_endpoints)를 확인해 보세요.

추론 엔드포인트는 API로 쉽게 접근할 수 있습니다. 이 엔드포인트들은 [Swagger](https://api.endpoints.huggingface.cloud/)를 통해 문서화되어 있고, [`InferenceEndpoint`] 클래스는 이 API를 사용해 만든 간단한 래퍼입니다.

## 매소드 [[methods]]

다음과 같은 추론 엔드포인트의 기능이 [`HfApi`]안에 구현되어 있습니다:

- [`get_inference_endpoint`]와 [`list_inference_endpoints`]를 사용해 엔드포인트 정보를 조회할 수 있습니다.
- [`create_inference_endpoint`], [`update_inference_endpoint`], [`delete_inference_endpoint`]로 엔드포인트를 배포하고 관리할 수 있습니다.
- [`pause_inference_endpoint`]와 [`resume_inference_endpoint`]로 엔드포인트를 잠시 멈추거나 다시 시작할 수 있습니다.
- [`scale_to_zero_inference_endpoint`]로 엔드포인트의 복제본을 0개로 설정할 수 있습니다.

## InferenceEndpoint [[huggingface_hub.InferenceEndpoint]]

기본 데이터 클래스는 [`InferenceEndpoint`]입니다. 여기에는 구성 및 현재 상태를 가지고 있는 배포된 `InferenceEndpoint`에 대한 정보가 포함되어 있습니다. 배포 후에는 [`InferenceEndpoint.client`]와 [`InferenceEndpoint.async_client`]를 사용해 엔드포인트에서 추론 작업을 할 수 있고, 이때 [`InferenceClient`]와 [`AsyncInferenceClient`] 객체를 반환합니다.

[[autodoc]] InferenceEndpoint
  - from_raw
  - client
  - async_client
  - all

## InferenceEndpointStatus [[huggingface_hub.InferenceEndpointStatus]]

[[autodoc]] InferenceEndpointStatus

## InferenceEndpointType [[huggingface_hub.InferenceEndpointType]]

[[autodoc]] InferenceEndpointType

## InferenceEndpointError [[huggingface_hub.InferenceEndpointError]]

[[autodoc]] InferenceEndpointError
