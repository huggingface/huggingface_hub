# 추론 엔드포인트 [[inference-endpoints]]

Hugging Face가 관리하는 전용 및 자동 확장 인프라에서 모델을 쉽고 안전하게 배포할 수 있도록 해주는 프로덕션 솔루션이 바로 추론 엔드포인트입니다. 이 추론 엔드포인트는 [허브](https://huggingface.co/models)에 있는 모델을 기반으로 구축됩니다. 이 문서는 `huggingface_hub`와 추론 엔드포인트의 통합에 대한 참고자료입니다. 추론 엔드포인트 제품에 대한 자세한 정보는 [공식 문서](https://huggingface.co/docs/inference-endpoints/index)에서 확인할 수 있습니다.

<Tip>

프로그래밍 방식으로 추론 엔드포인트를 관리하는 방법을 배우고 싶다면 [관련 가이드](../guides/inference_endpoints)를 참고하세요.

</Tip>

추론 엔드포인트는 API를 통해 완전히 관리됩니다. 이 엔드포인트들은 [Swagger](https://api.endpoints.huggingface.cloud/)를 통해 문서화되어 있으며, [`InferenceEndpoint`] 클래스는 이 API를 기반으로 만들어진 간단한 래퍼입니다.

## 메서드 [[methods]]

[`HfApi`]는 추론 엔드포인트의 다음과 같은 기능들을 제공합니다:

- [`get_inference_endpoint`]와 [`list_inference_endpoints`]로 추론 엔드포인트에 대한 정보를 조회합니다.
- [`create_inference_endpoint`], [`update_inference_endpoint`], [`delete_inference_endpoint`]로 추론 엔드포인트를 배포하고 관리합니다.
- [`pause_inference_endpoint`]와 [`resume_inference_endpoint`]로 추론 엔드포인트를 일시 정지하거나 재개합니다.
- [`scale_to_zero_inference_endpoint`]로 엔드포인트를 수동으로 0개의 복제본으로 조정합니다.

## InferenceEndpoint [[huggingface_hub.InferenceEndpoint]]

주 데이터 클래스인 [`InferenceEndpoint`]는 배포된 `InferenceEndpoint`에 대한 정보를 담고 있습니다. 여기에는 구성과 현재 상태 등이 포함됩니다. 배포가 완료된 후에는 [`InferenceEndpoint.client`]와 [`InferenceEndpoint.async_client`] 속성을 통해 엔드포인트에서 추론을 수행할 수 있으며, 각각 [`InferenceClient`]와 [`AsyncInferenceClient`] 객체를 반환합니다.

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
