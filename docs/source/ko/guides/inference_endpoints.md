# 추론 엔드포인트[[inference-endpoints]]

추론 엔드포인트는 Hugging Face가 관리하는 전용 및 자동 확장 인프라에 `transformers`, `sentence-transformers` 및 `diffusers` 모델을 쉽게 배포할 수 있는 안전한 프로덕션 솔루션을 제공합니다. 추론 엔드포인트는 [Hub](https://huggingface.co/models)의 모델로 구축됩니다.
이 가이드에서는 `huggingface_hub`를 사용하여 프로그래밍 방식으로 추론 엔드포인트를 관리하는 방법을 배웁니다. 추론 엔드포인트 제품 자체에 대한 자세한 내용은 [공식 문서](https://huggingface.co/docs/inference-endpoints/index)를 참조하세요.

이 가이드에서는 `huggingface_hub`가 올바르게 설치 및 로그인되어 있다고 가정합니다. 아직 그렇지 않은 경우 [빠른 시작 가이드](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart)를 참조하세요. 추론 엔드포인트 API를 지원하는 최소 버전은 `v0.19.0`입니다.

## 추론 엔드포인트 생성[[create-an-inference-endpoint]]

첫 번째 단계는 [`create_inference_endpoint`]를 사용하여 추론 엔드포인트를 생성하는 것입니다:

```py
>>> from huggingface_hub import create_inference_endpoint

>>> endpoint = create_inference_endpoint(
...     "my-endpoint-name",
...     repository="gpt2",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="cpu",
...     vendor="aws",
...     region="us-east-1",
...     type="protected",
...     instance_size="x2",
...     instance_type="intel-icl"
... )
```

예시에서는 `"my-endpoint-name"`라는 `protected` 추론 엔드포인트를 생성하여 `text-generation`을 위한 [gpt2](https://huggingface.co/gpt2)를 제공합니다. `protected` 추론 엔드포인트 API에 액세스하려면 토큰이 필요합니다. 또한 벤더, 지역, 액셀러레이터, 인스턴스 유형, 크기와 같은 하드웨어 요구 사항을 구성하기 위한 추가 정보를 제공해야 합니다. 사용 가능한 리소스 목록은 [여기](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors)에서 확인할 수 있습니다. 또한 [웹 인터페이스](https://ui.endpoints.huggingface.co/new)를 사용하여 편리하게 수동으로 추론 엔드포인트를 생성할 수 있습니다. 고급 설정 및 사용법에 대한 자세한 내용은 [이 가이드](https://huggingface.co/docs/inference-endpoints/guides/advanced)를 참조하세요.

[`create_inference_endpoint`]에서 반환된 값은 [`InferenceEndpoint`] 개체입니다:

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

이것은 엔드포인트에 대한 정보를 저장하는 데이터클래스입니다. `name`, `repository`, `status`, `task`, `created_at`, `updated_at` 등과 같은 중요한 속성에 접근할 수 있습니다. 필요한 경우 `endpoint.raw`를 통해 서버로부터의 원시 응답에도 접근할 수 있습니다.

추론 엔드포인트가 생성되면 [개인 대시보드](https://ui.endpoints.huggingface.co/)에서 확인할 수 있습니다.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)

#### 사용자 정의 이미지 사용[[using-a-custom-image]]

기본적으로 추론 엔드포인트는 Hugging Face에서 제공하는 도커 이미지로 구축됩니다. 그러나 `custom_image` 매개변수를 사용하여 모든 도커 이미지를 지정할 수 있습니다. 일반적인 사용 사례는 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 프레임워크를 사용하여 LLM을 실행하는 것입니다. 다음과 같이 수행할 수 있습니다:

```python
# TGI에서 Zephyr-7b-beta를 실행하는 추론 엔드포인트 시작하기
>>> from huggingface_hub import create_inference_endpoint
>>> endpoint = create_inference_endpoint(
...     "aws-zephyr-7b-beta-0486",
...     repository="HuggingFaceH4/zephyr-7b-beta",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="gpu",
...     vendor="aws",
...     region="us-east-1",
...     type="protected",
...     instance_size="x1",
...     instance_type="nvidia-a10g",
...     custom_image={
...         "health_route": "/health",
...         "env": {
...             "MAX_BATCH_PREFILL_TOKENS": "2048",
...             "MAX_INPUT_LENGTH": "1024",
...             "MAX_TOTAL_TOKENS": "1512",
...             "MODEL_ID": "/repository"
...         },
...         "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
...     },
... )
```

`custom_image`에 전달할 값은 도커 컨테이너의 URL과 이를 실행하기 위한 구성이 포함된 딕셔너리입니다. 자세한 내용은 [Swagger 문서](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint)를 참조하세요.

### 기존 추론 엔드포인트 가져오기 또는 리스트 조회[[get-or-list-existing-inference-endpoints]]

경우에 따라 이전에 생성한 추론 엔드포인트를 관리해야 할 수 있습니다. 이름을 알고 있는 경우 [`get_inference_endpoint`]를 사용하여 [`InferenceEndpoint`] 개체를 가져올 수 있습니다. 또는 [`list_inference_endpoints`]를 사용하여 모든 추론 엔드포인트 리스트를 검색할 수 있습니다. 두 메소드 모두 선택적 `namespace` 매개변수를 허용합니다. 속해 있는 조직의 `namespace`를 설정할 수 있습니다. 그렇지 않으면 기본적으로 사용자 이름이 사용됩니다.

```py
>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints

# 엔드포인트 개체 가져오기
>>> get_inference_endpoint("my-endpoint-name")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# 조직의 모든 추론 엔드포인트 나열
>>> list_inference_endpoints(namespace="huggingface")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]

# 사용자가 속해있는 모든 조직의 엔드포인트 나열
>>> list_inference_endpoints(namespace="*")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]
```

## 배포 상태 확인[[check-deployment-status]]

이 가이드의 나머지 부분에서는 `endpoint`라는 이름의 [`InferenceEndpoint`] 객체를 가지고 있다고 가정합니다. 엔드포인트에 `status` 속성이 [`InferenceEndpointStatus`] 유형이라는 것을 알 수 있었습니다. 추론 엔드포인트가 배포되고 접근 가능하면 상태가 `"running"`이 되고 `url` 속성이 설정됩니다:

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

`추론 엔드포인트가 "running"` 상태에 도달하기 전에 일반적으로 `"initializing"` 또는 `"pending"` 단계를 거칩니다. [`~InferenceEndpoint.fetch`]를 실행하여 엔드포인트의 새로운 상태를 가져올 수 있습니다. [`InferenceEndpoint`]의 다른 메소드와 마찬가지로 이 메소드는 서버에 요청을 하며, `endpoint`의 내부 속성이 변경됩니다:

```py
>>> endpoint.fetch()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

추론 엔드포인트가 실행될 때까지 기다리면서 상태를 가져오는 대신 [`~InferenceEndpoint.wait`]를 직접 호출할 수 있습니다. 이 헬퍼는 `timeout`과 `fetch_every` 매개변수를 입력으로 받아 (초 단위) 추론 엔드포인트가 배포될 때까지 스레드를 차단합니다. 기본값은 각각 `None`(제한 시간 없음)과 `5`초입니다.

```py
# 엔드포인트 보류
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# 10초 대기 => InferenceEndpointTimeoutError 발생
>>> endpoint.wait(timeout=10)
    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
huggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.

# 추가 대기
>>> endpoint.wait()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

`timeout`이 설정되어 있고 추론 엔드포인트를 불러오는 데 너무 오래 걸리면, [`InferenceEndpointTimeoutError`] 제한 시간 초과 오류가 발생합니다.

## 추론 실행[[run-inference]]

추론 엔드포인트가 실행되면, 마침내 추론을 실행할 수 있습니다!

[`InferenceEndpoint`]에는 각각 [`InferenceClient`]와 [`AsyncInferenceClient`]를 반환하는 `client`와 `async_client` 속성이 있습니다.

```py
# 텍스트 생성 작업 실행:
>>> endpoint.client.text_generation("I am")
' not a fan of the idea of a "big-budget" movie. I think it\'s a'

# 비동기 컨텍스트에서도 마찬가지로 실행:
>>> await endpoint.async_client.text_generation("I am")
```

추론 엔드포인트가 실행 중이 아니면 [`InferenceEndpointError`] 오류가 발생합니다:

```py
>>> endpoint.client
huggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.
```

[`InferenceClient`]를 사용하는 방법에 대한 자세한 내용은 [추론 가이드](../guides/inference)를 참조하세요.

## 라이프사이클 관리[[manage-lifecycle]]

이제 추론 엔드포인트를 생성하고 추론을 실행하는 방법을 살펴보았으니, 라이프사이클을 관리하는 방법을 살펴봅시다.

<Tip>

이 섹션에서는 [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] 및 [`~InferenceEndpoint.delete`] 등의 메소드를 살펴볼 것입니다. 모든 메소드는 편의를 위해 [`InferenceEndpoint`]에 추가된 별칭입니다. 원한다면 `HfApi`에 정의된 일반 메소드 [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`] 및 [`delete_inference_endpoint`]를 사용할 수도 있습니다.

</Tip>

### 일시 중지 또는 0으로 확장[[pause-or-scale-to-zero]]

추론 엔드포인트를 사용하지 않을 때 비용을 절감하기 위해 [`~InferenceEndpoint.pause`]를 사용하여 일시 중지하거나 [`~InferenceEndpoint.scale_to_zero`]를 사용하여 0으로 스케일링할 수 있습니다.

<Tip>

*일시 중지* 또는 *0으로 스케일링*된 추론 엔드포인트는 비용이 들지 않습니다. 이 두 가지의 차이점은 *일시 중지* 엔드포인트는 [`~InferenceEndpoint.resume`]를 사용하여 명시적으로 *재개*해야 한다는 것입니다. 반대로 *0으로 스케일링*된 엔드포인트는 추론 호출이 있으면 추가 콜드 스타트 지연과 함께 자동으로 시작됩니다. 추론 엔드포인트는 일정 기간 비활성화된 후 자동으로 0으로 스케일링되도록 구성할 수도 있습니다.

</Tip>

```py
# 엔드포인트 일시중지 및 재시작
>>> endpoint.pause()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)
>>> endpoint.resume()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
>>> endpoint.wait().client.text_generation(...)
...

# 0으로 스케일링
>>> endpoint.scale_to_zero()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
# 엔드포인트는 'running'은 아니지만 URL�을 가지고 있으며 첫 번째 호출 시 다시 시작됩니다.
```

### 모델 또는 하드웨어 요구 사항 업데이트[[update-model-or-hardware-requirements]]

경우에 따라 새로운 엔드포인트를 생성하지 않고 추론 엔드포인트를 업데이트하고 싶을 수 있습니다. 호스팅된 모델이나 모델 실행에 필요한 하드웨어 요구 사항을 업데이트할 수 있습니다. 이렇게 하려면 [`~InferenceEndpoint.update`]를 사용합니다:

```py
# 타겟 모델 변경
>>> endpoint.update(repository="gpt2-large")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# 복제본 갯수 업데이트
>>> endpoint.update(min_replica=2, max_replica=6)
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# 더 큰 인스턴스로 업데이트
>>> endpoint.update(accelerator="cpu", instance_size="x4", instance_type="intel-icl")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)
```

### 엔드포인트 삭제[[delete-the-endpoint]]

마지막으로 더 이상 추론 엔드포인트를 사용하지 않을 경우, [`~InferenceEndpoint.delete()`]를 호출하기만 하면 됩니다.

<Tip warning={true}>

이것은 돌이킬 수 없는 작업이며, 구성, 로그 및 사용 메트릭을 포함한 엔드포인트를 완전히 제거합니다. 삭제된 추론 엔드포인트는 복원할 수 없습니다.

</Tip>

## 엔드 투 엔드 예제[an-end-to-end-example]

추론 엔드포인트의 일반적인 사용 사례는 한 번에 여러 개의 작업을 처리하여 인프라 비용을 제한하는 것입니다. 이 가이드에서 본 것을 사용하여 이 프로세스를 자동화할 수 있습니다:

```py
>>> import asyncio
>>> from huggingface_hub import create_inference_endpoint

# 엔드포인트 시작 + 초기화될 때까지 대기
>>> endpoint = create_inference_endpoint(name="batch-endpoint",...).wait()

# 추론 실행
>>> client = endpoint.client
>>> results = [client.text_generation(...) for job in jobs]

# 비동기 추론 실행
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# 엔드포인트 중지
>>> endpoint.pause()
```

또는 추론 엔드포인트가 이미 존재하고 일시 중지된 경우:

```py
>>> import asyncio
>>> from huggingface_hub import get_inference_endpoint

# 엔드포인트 가져오기 + 초기화될 때까지 대기
>>> endpoint = get_inference_endpoint("batch-endpoint").resume().wait()

# 추론 실행
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# 엔드포인트 중지
>>> endpoint.pause()
```
