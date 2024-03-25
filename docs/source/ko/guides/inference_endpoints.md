# 추론 엔드포인트

추론 엔드포인트는 쉽게 `transformers`, `sentence-transformers` 및 `diffusers` 모델을 자동 스케일링되는 전용 인프라에 배포할 수 있는 안전한 프로덕션 솔루션을 제공합니다. 추론 엔드포인트는 [Hub](https://huggingface.co/models)의 모델에서 구축됩니다.
이 가이드에서는 `huggingface_hub`를 사용하여 프로그래밍 방식으로 추론 엔드포인트를 관리하는 방법을 배웁니다. 추론 엔드포인트 제품 자체에 대한 자세한 내용은 [공식 문서](https://huggingface.co/docs/inference-endpoints/index)를 확인하세요.

이 가이드는 `huggingface_hub`가 올바르게 설치되고 기계가 로그인되어 있다고 가정합니다. 아직 그렇지 않은 경우 [빠른 시작 가이드](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart)를 확인하세요. 추론 엔드포인트 API를 지원하는 최소 버전은 `v0.19.0`입니다.

## 추론 엔드포인트 생성

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
...     instance_size="medium",
...     instance_type="c6i"
... )
```

이 예제에서는 `"gpt2"`를 서비스하기 위해 `"my-endpoint-name"`이라는 `protected` 추론 엔드포인트를 생성했습니다. `protected` 추론 엔드포인트는 토큰이 API에 액세스하는 데 필요하다는 의미입니다. 하드웨어 요구 사항 구성을 위해 공급업체, 지역, 가속기, 인스턴스 유형 및 크기와 같은 추가 정보를 제공해야 합니다. 사용 가능한 리소스 목록은 [여기](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors)에서 확인할 수 있습니다. 또는 편의를 위해 [웹 인터페이스](https://ui.endpoints.huggingface.co/new)를 사용하여 수동으로 추론 엔드포인트를 만들 수 있습니다. 고급 설정 및 사용에 대한 자세한 내용은 [이 가이드](https://huggingface.co/docs/inference-endpoints/guides/advanced)를 참조하세요.

[`create_inference_endpoint`]에 의해 반환된 값은 [`InferenceEndpoint`] 객체입니다.

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

이것은 엔드포인트에 대한 정보를 보유하는 데이터 클래스입니다. `name`, `repository`, `status`, `task`, `created_at`, `updated_at` 등의 중요한 속성에 액세스할 수 있습니다. 필요한 경우 `endpoint.raw`를 사용하여 서버의 원시 응답에 액세스할 수도 있습니다.

추론 엔드포인트를 생성하면 [개인 대시보드](https://ui.endpoints.huggingface.co/)에서 찾을 수 있습니다.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)

#### 사용자 지정 이미지 사용

기본적으로 추론 엔드포인트는 Hugging Face에서 제공하는 도커 이미지에서 빌드됩니다. 그러나 `custom_image` 매개변수를 사용하여 모든 도커 이미지를 지정할 수 있습니다. 일반적인 사용 사례는 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 프레임워크를 사용하여 LLM을 실행하는 것입니다. 다음과 같이 할 수 있습니다:

```python 
# TGI에서 Zephyr-7b-beta 실행 시작
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
...     instance_size="medium",
...     instance_type="g5.2xlarge",
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

`custom_image`로 전달할 값은 도커 컨테이너의 URL과 이를 실행하기 위한 구성이 포함된 딕셔너리입니다. 자세한 내용은 [Swagger 문서](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint)를 확인하세요.

### 기존 추론 엔드포인트 가져오기 또는 나열하기

경우에 따라 이전에 생성한 추론 엔드포인트를 관리해야 할 수 있습니다. 이름을 알고 있다면 [`get_inference_endpoint`]를 사용하여 가져올 수 있으며, 이는 [`InferenceEndpoint`] 객체를 반환합니다. 또는 [`list_inference_endpoints`]를 사용하여 모든 추론 엔드포인트 목록을 검색할 수 있습니다. 두 메서드 모두 선택적 `namespace` 매개변수를 허용합니다. 자신이 속한 모든 조직의 `namespace`를 설정할 수 있습니다. 그렇지 않으면 기본값은 사용자 이름입니다.

```py
>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints

# 하나 가져오기
>>> get_inference_endpoint("my-endpoint-name")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# 조직의 모든 엔드포인트 나열
>>> list_inference_endpoints(namespace="huggingface")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]

# 사용자가 속한 모든 조직의 모든 엔드포인트 나열
>>> list_inference_endpoints(namespace="*")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]
```

## 배포 상태 확인

이 가이드의 나머지 부분에서는 `endpoint`라는 [`InferenceEndpoint`] 객체를 갖고 있다고 가정합니다. 엔드포인트에 [`InferenceEndpointStatus`] 유형의 `status` 속성이 있음을 알았을 것입니다. 추론 엔드포인트가 배포되고 액세스 가능해지면 상태는 `"running"`이 되고 `url` 속성이 설정됩니다.

```py
>>> endpoint  
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

`"running"` 상태에 도달하기 전에 추론 엔드포인트는 일반적으로 `"initializing"` 또는 `"pending"` 단계를 거칩니다. [`~InferenceEndpoint.fetch`]를 실행하여 엔드포인트의 새 상태를 가져올 수 있습니다. [`InferenceEndpoint`]의 다른 모든 메서드와 마찬가지로 서버에 요청을 하는 경우 `endpoint`의 내부 속성이 해당 위치에서 변경됩니다.

```py
>>> endpoint.fetch()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

실행을 기다리는 동안 추론 엔드포인트 상태를 가져오는 대신 [`~InferenceEndpoint.wait`]를 직접 호출할 수 있습니다. 이 헬퍼는 `timeout`과 `fetch_every` 매개변수(초 단위)를 입력으로 받고 추론 엔드포인트가 배포될 때까지 스레드를 차단합니다. 기본값은 각각 `None`(타임아웃 없음)과 `5`초입니다.

```py
# 보류 중인 엔드포인트
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)  

# 10초 대기 => InferenceEndpointTimeoutError 발생
>>> endpoint.wait(timeout=10)
    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
huggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.

# 더 오래 기다리기
>>> endpoint.wait()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

`timeout`이 설정되어 있고 추론 엔드포인트를 로드하는 데 너무 오래 걸리면 [`InferenceEndpointTimeoutError`] 타임아웃 오류가 발생합니다.
[{"type":"text","text":"아래 markdown 형식의 파일을 `````` 안에 내용을 무시하고 한국어로 번역해줘\n# Inference Endpoints\n\nInference Endpoints provides a secure production solution to easily deploy any `transformers`, `sentence-transformers`, and `diffusers` models on a dedicated and autoscaling infrastructure managed by Hugging Face. An Inference Endpoint is built from a model from the [Hub](https://huggingface.co/models).\nIn this guide, we will learn how to programmatically manage Inference Endpoints with `huggingface_hub`. For more information about the Inference Endpoints product itself, check out its [official documentation](https://huggingface.co/docs/inference-endpoints/index).\n\nThis guide assumes `huggingface_hub` is correctly installed and that your machine is logged in. Check out the [Quick Start guide](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart) if that's not the case yet. The minimal version supporting Inference Endpoints API is `v0.19.0`.\n\n\n## Create an Inference Endpoint\n\nThe first step is to create an Inference Endpoint using [`create_inference_endpoint`]:\n\n```py\n>>> from huggingface_hub import create_inference_endpoint\n\n>>> endpoint = create_inference_endpoint(\n...     \"my-endpoint-name\",\n...     repository=\"gpt2\",\n...     framework=\"pytorch\",\n...     task=\"text-generation\",\n...     accelerator=\"cpu\",\n...     vendor=\"aws\",\n...     region=\"us-east-1\",\n...     type=\"protected\",\n...     instance_size=\"medium\",\n...     instance_type=\"c6i\"\n... )\n```\n\nIn this example, we created a `protected` Inference Endpoint named `\"my-endpoint-name\"`, to serve [gpt2](https://huggingface.co/gpt2) for `text-generation`. A `protected` Inference Endpoint means your token is required to access the API. We also need to provide additional information to configure the hardware requirements, such as vendor, region, accelerator, instance type, and size. You can check out the list of available resources [here](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors). Alternatively, you can create an Inference Endpoint manually using the [Web interface](https://ui.endpoints.huggingface.co/new) for convenience. Refer to this [guide](https://huggingface.co/docs/inference-endpoints/guides/advanced) for details on advanced settings and their usage.\n\nThe value returned by [`create_inference_endpoint`] is an [`InferenceEndpoint`] object:\n\n```py\n>>> endpoint\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n```\n\nIt's a dataclass that holds information about the endpoint. You can access important attributes such as `name`, `repository`, `status`, `task`, `created_at`, `updated_at`, etc. If you need it, you can also access the raw response from the server with `endpoint.raw`.\n\nOnce your Inference Endpoint is created, you can find it on your [personal dashboard](https://ui.endpoints.huggingface.co/).\n\n![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)\n\n#### Using a custom image\n\nBy default the Inference Endpoint is built from a docker image provided by Hugging Face. However, it is possible to specify any docker image using the `custom_image` parameter. A common use case is to run LLMs using the [text-generation-inference](https://github.com/huggingface/text-generation-inference) framework. This can be done like this:\n\n```python\n# Start an Inference Endpoint running Zephyr-7b-beta on TGI\n>>> from huggingface_hub import create_inference_endpoint\n>>> endpoint = create_inference_endpoint(\n...     \"aws-zephyr-7b-beta-0486\",\n...     repository=\"HuggingFaceH4/zephyr-7b-beta\",\n...     framework=\"pytorch\",\n...     task=\"text-generation\",\n...     accelerator=\"gpu\",\n...     vendor=\"aws\",\n...     region=\"us-east-1\",\n...     type=\"protected\",\n...     instance_size=\"medium\",\n...     instance_type=\"g5.2xlarge\",\n...     custom_image={\n...         \"health_route\": \"/health\",\n...         \"env\": {\n...             \"MAX_BATCH_PREFILL_TOKENS\": \"2048\",\n...             \"MAX_INPUT_LENGTH\": \"1024\",\n...             \"MAX_TOTAL_TOKENS\": \"1512\",\n...             \"MODEL_ID\": \"/repository\"\n...         },\n...         \"url\": \"ghcr.io/huggingface/text-generation-inference:1.1.0\",\n...     },\n... )\n```\n\nThe value to pass as `custom_image` is a dictionary containing a url to the docker container and configuration to run it. For more details about it, checkout the [Swagger documentation](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint).\n\n### Get or list existing Inference Endpoints\n\nIn some cases, you might need to manage Inference Endpoints you created previously. If you know the name, you can fetch it using [`get_inference_endpoint`], which returns an [`InferenceEndpoint`] object. Alternatively, you can use [`list_inference_endpoints`] to retrieve a list of all Inference Endpoints. Both methods accept an optional `namespace` parameter. You can set the `namespace` to any organization you are a part of. Otherwise, it defaults to your username.\n\n```py\n>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints\n\n# Get one\n>>> get_inference_endpoint(\"my-endpoint-name\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n\n# List all endpoints from an organization\n>>> list_inference_endpoints(namespace=\"huggingface\")\n[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]\n\n# List all endpoints from all organizations the user belongs to\n>>> list_inference_endpoints(namespace=\"*\")\n[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]\n```\n\n## Check deployment status\n\nIn the rest of this guide, we will assume that we have a [`InferenceEndpoint`] object called `endpoint`. You might have noticed that the endpoint has a `status` attribute of type [`InferenceEndpointStatus`]. When the Inference Endpoint is deployed and accessible, the status should be `\"running\"` and the `url` attribute is set:\n\n```py\n>>> endpoint\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n```\n\nBefore reaching a `\"running\"` state, the Inference Endpoint typically goes through an `\"initializing\"` or `\"pending\"` phase. You can fetch the new state of the endpoint by running [`~InferenceEndpoint.fetch`]. Like every other method from [`InferenceEndpoint`] that makes a request to the server, the internal attributes of `endpoint` are mutated in place:\n\n```py\n>>> endpoint.fetch()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n```\n\nInstead of fetching the Inference Endpoint status while waiting for it to run, you can directly call [`~InferenceEndpoint.wait`]. This helper takes as input a `timeout` and a `fetch_every` parameter (in seconds) and will block the thread until the Inference Endpoint is deployed. Default values are respectively `None` (no timeout) and `5` seconds.\n\n```py\n# Pending endpoint\n>>> endpoint\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n\n# Wait 10s => raises a InferenceEndpointTimeoutError\n>>> endpoint.wait(timeout=10)\n    raise InferenceEndpointTimeoutError(\"Timeout while waiting for Inference Endpoint to be deployed.\")\nhuggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.\n\n# Wait more\n>>> endpoint.wait()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n```\n\nIf `timeout` is set and the Inference Endpoint takes too much time to load, a [`InferenceEndpointTimeoutError`] timeout error is raised.\n\n## Run inference\n\nOnce your Inference Endpoint is up and running, you can finally run inference on it!\n\n[`InferenceEndpoint`] has two properties `client` and `async_client` returning respectively an [`InferenceClient`] and an [`AsyncInferenceClient`] objects.\n\n```py\n# Run text_generation task:\n>>> endpoint.client.text_generation(\"I am\")\n' not a fan of the idea of a \"big-budget\" movie. I think it\\'s a'\n\n# Or in an asyncio context:\n>>> await endpoint.async_client.text_generation(\"I am\")\n```\n\nIf the Inference Endpoint is not running, an [`InferenceEndpointError`] exception is raised:\n\n```py\n>>> endpoint.client\nhuggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.\n```\n\nFor more details about how to use the [`InferenceClient`], check out the [Inference guide](../guides/inference).\n\n## Manage lifecycle\n\nNow that we saw how to create an Inference Endpoint and run inference on it, let's see how to manage its lifecycle.\n\n<Tip>\n\nIn this section, we will see methods like [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] and [`~InferenceEndpoint.delete`]. All of those methods are aliases added to [`InferenceEndpoint`] for convenience. If you prefer, you can also use the generic methods defined in `HfApi`: [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`], and [`delete_inference_endpoint`].\n\n</Tip>\n\n### Pause or scale to zero\n\nTo reduce costs when your Inference Endpoint is not in use, you can choose to either pause it using [`~InferenceEndpoint.pause`] or scale it to zero using [`~InferenceEndpoint.scale_to_zero`].\n\n<Tip>\n\nAn Inference Endpoint that is *paused* or *scaled to zero* doesn't cost anything. The difference between those two is that a *paused* endpoint needs to be explicitly *resumed* using [`~InferenceEndpoint.resume`]. On the contrary, a *scaled to zero* endpoint will automatically start if an inference call is made to it, with an additional cold start delay. An Inference Endpoint can also be configured to scale to zero automatically after a certain period of inactivity.\n\n</Tip>\n\n```py\n# Pause and resume endpoint\n>>> endpoint.pause()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)\n>>> endpoint.resume()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n>>> endpoint.wait().client.text_generation(...)\n...\n\n# Scale to zero\n>>> endpoint.scale_to_zero()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n# Endpoint is not 'running' but still has a URL and will restart on first call.\n```\n\n### Update model or hardware requirements\n\nIn some cases, you might also want to update your Inference Endpoint without creating a new one. You can either update the hosted model or the hardware requirements to run the model. You can do this using [`~InferenceEndpoint.update`]:\n\n```py\n# Change target model\n>>> endpoint.update(repository=\"gpt2-large\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n\n# Update number of replicas\n>>> endpoint.update(min_replica=2, max_replica=6)\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n\n# Update to larger instance\n>>> endpoint.update(accelerator=\"cpu\", instance_size=\"large\", instance_type=\"c6i\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n```\n\n### Delete the endpoint\n\nFinally if you won't use the Inference Endpoint anymore, you can simply call [`~InferenceEndpoint.delete()`].\n\n<Tip warning={true}>\n\nThis is a non-revertible action that will completely remove the endpoint, including its configuration, logs and usage metrics. You cannot restore a deleted Inference Endpoint.\n\n</Tip>\n\n\n## An end-to-end example\n\nA typical use case of Inference Endpoints is to process a batch of jobs at once to limit the infrastructure costs. You can automate this process using what we saw in this guide:\n\n```py\n>>> import asyncio\n>>> from huggingface_hub import create_inference_endpoint\n\n# Start endpoint + wait until initialized\n>>> endpoint = create_inference_endpoint(name=\"batch-endpoint\",...).wait()\n\n# Run inference\n>>> client = endpoint.client\n>>> results = [client.text_generation(...) for job in jobs]\n\n# Or with asyncio\n>>> async_client = endpoint.async_client\n>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])\n\n# Pause endpoint\n>>> endpoint.pause()\n```\n\nOr if your Inference Endpoint already exists and is paused:\n\n```py\n>>> import asyncio\n>>> from huggingface_hub import get_inference_endpoint\n\n# Get endpoint + wait until initialized\n>>> endpoint = get_inference_endpoint(\"batch-endpoint\").resume().wait()\n\n# Run inference\n>>> async_client = endpoint.async_client\n>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])\n\n# Pause endpoint\n>>> endpoint.pause()\n```\n"}]
# 추론 엔드포인트

추론 엔드포인트는 쉽게 `transformers`, `sentence-transformers` 및 `diffusers` 모델을 자동 스케일링되는 전용 인프라에 배포할 수 있는 안전한 프로덕션 솔루션을 제공합니다. 추론 엔드포인트는 [Hub](https://huggingface.co/models)의 모델에서 구축됩니다.
이 가이드에서는 `huggingface_hub`를 사용하여 프로그래밍 방식으로 추론 엔드포인트를 관리하는 방법을 배웁니다. 추론 엔드포인트 제품 자체에 대한 자세한 내용은 [공식 문서](https://huggingface.co/docs/inference-endpoints/index)를 확인하세요.

이 가이드는 `huggingface_hub`가 올바르게 설치되고 기계가 로그인되어 있다고 가정합니다. 아직 그렇지 않은 경우 [빠른 시작 가이드](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart)를 확인하세요. 추론 엔드포인트 API를 지원하는 최소 버전은 `v0.19.0`입니다.

## 추론 엔드포인트 생성

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
...     instance_size="medium",
...     instance_type="c6i"
... )
```

이 예제에서는 `"gpt2"`를 서비스하기 위해 `"my-endpoint-name"`이라는 `protected` 추론 엔드포인트를 생성했습니다. `protected` 추론 엔드포인트는 토큰이 API에 액세스하는 데 필요하다는 의미입니다. 하드웨어 요구 사항 구성을 위해 공급업체, 지역, 가속기, 인스턴스 유형 및 크기와 같은 추가 정보를 제공해야 합니다. 사용 가능한 리소스 목록은 [여기](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors)에서 확인할 수 있습니다. 또는 편의를 위해 [웹 인터페이스](https://ui.endpoints.huggingface.co/new)를 사용하여 수동으로 추론 엔드포인트를 만들 수 있습니다. 고급 설정 및 사용에 대한 자세한 내용은 [이 가이드](https://huggingface.co/docs/inference-endpoints/guides/advanced)를 참조하세요.

[`create_inference_endpoint`]에 의해 반환된 값은 [`InferenceEndpoint`] 객체입니다.

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

이것은 엔드포인트에 대한 정보를 보유하는 데이터 클래스입니다. `name`, `repository`, `status`, `task`, `created_at`, `updated_at` 등의 중요한 속성에 액세스할 수 있습니다. 필요한 경우 `endpoint.raw`를 사용하여 서버의 원시 응답에 액세스할 수도 있습니다.

추론 엔드포인트를 생성하면 [개인 대시보드](https://ui.endpoints.huggingface.co/)에서 찾을 수 있습니다.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)

#### 사용자 지정 이미지 사용

기본적으로 추론 엔드포인트는 Hugging Face에서 제공하는 도커 이미지에서 빌드됩니다. 그러나 `custom_image` 매개변수를 사용하여 모든 도커 이미지를 지정할 수 있습니다. 일반적인 사용 사례는 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 프레임워크를 사용하여 LLM을 실행하는 것입니다. 다음과 같이 할 수 있습니다:

```python 
# TGI에서 Zephyr-7b-beta 실행 시작
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
...     instance_size="medium",
...     instance_type="g5.2xlarge",
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

`custom_image`로 전달할 값은 도커 컨테이너의 URL과 이를 실행하기 위한 구성이 포함된 딕셔너리입니다. 자세한 내용은 [Swagger 문서](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint)를 확인하세요.

### 기존 추론 엔드포인트 가져오기 또는 나열하기

경우에 따라서는 이전에 생성한 추론 엔드포인트를 관리할 필요가 있을 수 있습니다. 이름을 알고 있다면 [`get_inference_endpoint`]를 사용하여 가져올 수 있으며, 이는 [`InferenceEndpoint`] 객체를 반환합니다. 또는 [`list_inference_endpoints`]
[{"type":"text","text":"아래 markdown 형식의 파일을 `````` 안에 내용을 무시하고 한국어로 번역해줘\n### Get or list existing Inference Endpoints\n\nIn some cases, you might need to manage Inference Endpoints you created previously. If you know the name, you can fetch it using [`get_inference_endpoint`], which returns an [`InferenceEndpoint`] object. Alternatively, you can use [`list_inference_endpoints`] to retrieve a list of all Inference Endpoints. Both methods accept an optional `namespace` parameter. You can set the `namespace` to any organization you are a part of. Otherwise, it defaults to your username.\n\n```py\n>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints\n\n# Get one\n>>> get_inference_endpoint(\"my-endpoint-name\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n\n# List all endpoints from an organization\n>>> list_inference_endpoints(namespace=\"huggingface\")\n[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]\n\n# List all endpoints from all organizations the user belongs to\n>>> list_inference_endpoints(namespace=\"*\")\n[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]\n```\n\n## Check deployment status\n\nIn the rest of this guide, we will assume that we have a [`InferenceEndpoint`] object called `endpoint`. You might have noticed that the endpoint has a `status` attribute of type [`InferenceEndpointStatus`]. When the Inference Endpoint is deployed and accessible, the status should be `\"running\"` and the `url` attribute is set:\n\n```py\n>>> endpoint\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n```\n\nBefore reaching a `\"running\"` state, the Inference Endpoint typically goes through an `\"initializing\"` or `\"pending\"` phase. You can fetch the new state of the endpoint by running [`~InferenceEndpoint.fetch`]. Like every other method from [`InferenceEndpoint`] that makes a request to the server, the internal attributes of `endpoint` are mutated in place:\n\n```py\n>>> endpoint.fetch()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n```\n\nInstead of fetching the Inference Endpoint status while waiting for it to run, you can directly call [`~InferenceEndpoint.wait`]. This helper takes as input a `timeout` and a `fetch_every` parameter (in seconds) and will block the thread until the Inference Endpoint is deployed. Default values are respectively `None` (no timeout) and `5` seconds.\n\n```py\n# Pending endpoint\n>>> endpoint\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n\n# Wait 10s => raises a InferenceEndpointTimeoutError\n>>> endpoint.wait(timeout=10)\n    raise InferenceEndpointTimeoutError(\"Timeout while waiting for Inference Endpoint to be deployed.\")\nhuggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.\n\n# Wait more\n>>> endpoint.wait()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n```\n\nIf `timeout` is set and the Inference Endpoint takes too much time to load, a [`InferenceEndpointTimeoutError`] timeout error is raised.\n\n## Run inference\n\nOnce your Inference Endpoint is up and running, you can finally run inference on it!\n\n[`InferenceEndpoint`] has two properties `client` and `async_client` returning respectively an [`InferenceClient`] and an [`AsyncInferenceClient`] objects.\n\n```py\n# Run text_generation task:\n>>> endpoint.client.text_generation(\"I am\")\n' not a fan of the idea of a \"big-budget\" movie. I think it\\'s a'\n\n# Or in an asyncio context:\n>>> await endpoint.async_client.text_generation(\"I am\")\n```\n\nIf the Inference Endpoint is not running, an [`InferenceEndpointError`] exception is raised:\n\n```py\n>>> endpoint.client\nhuggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.\n```\n\nFor more details about how to use the [`InferenceClient`], check out the [Inference guide](../guides/inference).\n\n## Manage lifecycle\n\nNow that we saw how to create an Inference Endpoint and run inference on it, let's see how to manage its lifecycle.\n\n<Tip>\n\nIn this section, we will see methods like [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] and [`~InferenceEndpoint.delete`]. All of those methods are aliases added to [`InferenceEndpoint`] for convenience. If you prefer, you can also use the generic methods defined in `HfApi`: [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`], and [`delete_inference_endpoint`].\n\n</Tip>\n\n### Pause or scale to zero\n\nTo reduce costs when your Inference Endpoint is not in use, you can choose to either pause it using [`~InferenceEndpoint.pause`] or scale it to zero using [`~InferenceEndpoint.scale_to_zero`].\n\n<Tip>\n\nAn Inference Endpoint that is *paused* or *scaled to zero* doesn't cost anything. The difference between those two is that a *paused* endpoint needs to be explicitly *resumed* using [`~InferenceEndpoint.resume`]. On the contrary, a *scaled to zero* endpoint will automatically start if an inference call is made to it, with an additional cold start delay. An Inference Endpoint can also be configured to scale to zero automatically after a certain period of inactivity.\n\n</Tip>\n\n```py\n# Pause and resume endpoint\n>>> endpoint.pause()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)\n>>> endpoint.resume()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n>>> endpoint.wait().client.text_generation(...)\n...\n\n# Scale to zero\n>>> endpoint.scale_to_zero()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n# Endpoint is not 'running' but still has a URL and will restart on first call.\n```\n\n### Update model or hardware requirements\n\nIn some cases, you might also want to update your Inference Endpoint without creating a new one. You can either update the hosted model or the hardware requirements to run the model. You can do this using [`~InferenceEndpoint.update`]:\n\n```py\n# Change target model\n>>> endpoint.update(repository=\"gpt2-large\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n\n# Update number of replicas\n>>> endpoint.update(min_replica=2, max_replica=6)\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n\n# Update to larger instance\n>>> endpoint.update(accelerator=\"cpu\", instance_size=\"large\", instance_type=\"c6i\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n```\n\n### Delete the endpoint\n\nFinally if you won't use the Inference Endpoint anymore, you can simply call [`~InferenceEndpoint.delete()`].\n\n<Tip warning={true}>\n\nThis is a non-revertible action that will completely remove the endpoint, including its configuration, logs and usage metrics. You cannot restore a deleted Inference Endpoint.\n\n</Tip>\n\n\n## An end-to-end example\n\nA typical use case of Inference Endpoints is to process a batch of jobs at once to limit the infrastructure costs. You can automate this process using what we saw in this guide:\n\n```py\n>>> import asyncio\n>>> from huggingface_hub import create_inference_endpoint\n\n# Start endpoint + wait until initialized\n>>> endpoint = create_inference_endpoint(name=\"batch-endpoint\",...).wait()\n\n# Run inference\n>>> client = endpoint.client\n>>> results = [client.text_generation(...) for job in jobs]\n\n# Or with asyncio\n>>> async_client = endpoint.async_client\n>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])\n\n# Pause endpoint\n>>> endpoint.pause()\n```\n\nOr if your Inference Endpoint already exists and is paused:\n\n```py\n>>> import asyncio\n>>> from huggingface_hub import get_inference_endpoint\n\n# Get endpoint + wait until initialized\n>>> endpoint = get_inference_endpoint(\"batch-endpoint\").resume().wait()\n\n# Run inference\n>>> async_client = endpoint.async_client\n>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])\n\n# Pause endpoint\n>>> endpoint.pause()\n```\n"}]
### 기존 추론 엔드포인트 가져오기 또는 나열하기

경우에 따라 이전에 생성한 추론 엔드포인트를 관리해야 할 수 있습니다. 이름을 알고 있다면 [`get_inference_endpoint`]를 사용하여 가져올 수 있으며, 이는 [`InferenceEndpoint`] 객체를 반환합니다. 또는 [`list_inference_endpoints`]를 사용하여 모든 추론 엔드포인트 목록을 검색할 수 있습니다. 두 메서드 모두 선택적 `namespace` 매개변수를 허용합니다. 자신이 속한 모든 조직의 `namespace`를 설정할 수 있습니다. 그렇지 않으면 기본값은 사용자 이름입니다.

```py
>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints

# 하나 가져오기
>>> get_inference_endpoint("my-endpoint-name")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# 조직의 모든 엔드포인트 나열
>>> list_inference_endpoints(namespace="huggingface")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]

# 사용자가 속한 모든 조직의 모든 엔드포인트 나열
>>> list_inference_endpoints(namespace="*")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]
```

## 배포 상태 확인

이 가이드의 나머지 부분에서는 `endpoint`라는 [`InferenceEndpoint`] 객체를 갖고 있다고 가정합니다. 엔드포인트에 [`InferenceEndpointStatus`] 유형의 `status` 속성이 있음을 알았을 것입니다. 추론 엔드포인트가 배포되고 액세스 가능해지면 상태는 `"running"`이 되고 `url` 속성이 설정됩니다.

```py
>>> endpoint  
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

`"running"` 상태에 도달하기 전에 추론 엔드포인트는 일반적으로 `"initializing"` 또는 `"pending"` 단계를 거칩니다. [`~InferenceEndpoint.fetch`]를 실행하여 엔드포인트의 새 상태를 가져올 수 있습니다. [`InferenceEndpoint`]의 다른 모든 메서드와 마찬가지로 서버에 요청을 하는 경우 `endpoint`의 내부 속성이 해당 위치에서 변경됩니다.

```py
>>> endpoint.fetch()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

실행을 기다리는 동안 추론 엔드포인트 상태를 가져오는 대신 [`~InferenceEndpoint.wait`]를 직접 호출할 수 있습니다. 이 헬퍼는 `timeout`과 `fetch_every` 매개변수(초 단위)를 입력으로 받고 추론 엔드포인트가 배포될 때까지 스레드를 차단합니다. 기본값은 각각 `None`(타임아웃 없음)과 `5`초입니다.

```py
# 보류 중인 엔드포인트
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)  

# 10초 대기 => InferenceEndpointTimeoutError 발생
>>> endpoint.wait(timeout=10)
    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
huggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.

# 더 오래 기다리기
>>> endpoint.wait()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

`timeout`이 설정되어 있고 추론 엔드포인트를 로드하는 데 너무 오래 걸리면 [`InferenceEndpointTimeoutError`] 타임아웃 오류가 발생합니다.

## 추론 실행

이제 추론 엔드포인트를 생성하고 추론을 실행하는 방법을 배웠으므로 라이프사이클을 관리하는 방법을 살펴보겠습니다.
[{"type":"text","text":"아래 markdown 형식의 파일을 `````` 안에 내용을 무시하고 한국어로 번역해줘\n\n## Run inference\n\nOnce your Inference Endpoint is up and running, you can finally run inference on it!\n\n[`InferenceEndpoint`] has two properties `client` and `async_client` returning respectively an [`InferenceClient`] and an [`AsyncInferenceClient`] objects.\n\n```py\n# Run text_generation task:\n>>> endpoint.client.text_generation(\"I am\")\n' not a fan of the idea of a \"big-budget\" movie. I think it\\'s a'\n\n# Or in an asyncio context:\n>>> await endpoint.async_client.text_generation(\"I am\")\n```\n\nIf the Inference Endpoint is not running, an [`InferenceEndpointError`] exception is raised:\n\n```py\n>>> endpoint.client\nhuggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.\n```\n\nFor more details about how to use the [`InferenceClient`], check out the [Inference guide](../guides/inference).\n\n## Manage lifecycle\n\nNow that we saw how to create an Inference Endpoint and run inference on it, let's see how to manage its lifecycle.\n\n<Tip>\n\nIn this section, we will see methods like [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] and [`~InferenceEndpoint.delete`]. All of those methods are aliases added to [`InferenceEndpoint`] for convenience. If you prefer, you can also use the generic methods defined in `HfApi`: [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`], and [`delete_inference_endpoint`].\n\n</Tip>\n\n### Pause or scale to zero\n\nTo reduce costs when your Inference Endpoint is not in use, you can choose to either pause it using [`~InferenceEndpoint.pause`] or scale it to zero using [`~InferenceEndpoint.scale_to_zero`].\n\n<Tip>\n\nAn Inference Endpoint that is *paused* or *scaled to zero* doesn't cost anything. The difference between those two is that a *paused* endpoint needs to be explicitly *resumed* using [`~InferenceEndpoint.resume`]. On the contrary, a *scaled to zero* endpoint will automatically start if an inference call is made to it, with an additional cold start delay. An Inference Endpoint can also be configured to scale to zero automatically after a certain period of inactivity.\n\n</Tip>\n\n```py\n# Pause and resume endpoint\n>>> endpoint.pause()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)\n>>> endpoint.resume()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)\n>>> endpoint.wait().client.text_generation(...)\n...\n\n# Scale to zero\n>>> endpoint.scale_to_zero()\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')\n# Endpoint is not 'running' but still has a URL and will restart on first call.\n```\n\n### Update model or hardware requirements\n\nIn some cases, you might also want to update your Inference Endpoint without creating a new one. You can either update the hosted model or the hardware requirements to run the model. You can do this using [`~InferenceEndpoint.update`]:\n\n```py\n# Change target model\n>>> endpoint.update(repository=\"gpt2-large\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n\n# Update number of replicas\n>>> endpoint.update(min_replica=2, max_replica=6)\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n\n# Update to larger instance\n>>> endpoint.update(accelerator=\"cpu\", instance_size=\"large\", instance_type=\"c6i\")\nInferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)\n```\n\n### Delete the endpoint\n\nFinally if you won't use the Inference Endpoint anymore, you can simply call [`~InferenceEndpoint.delete()`].\n\n<Tip warning={true}>\n\nThis is a non-revertible action that will completely remove the endpoint, including its configuration, logs and usage metrics. You cannot restore a deleted Inference Endpoint.\n\n</Tip>\n\n\n## An end-to-end example\n\nA typical use case of Inference Endpoints is to process a batch of jobs at once to limit the infrastructure costs. You can automate this process using what we saw in this guide:\n\n```py\n>>> import asyncio\n>>> from huggingface_hub import create_inference_endpoint\n\n# Start endpoint + wait until initialized\n>>> endpoint = create_inference_endpoint(name=\"batch-endpoint\",...).wait()\n\n# Run inference\n>>> client = endpoint.client\n>>> results = [client.text_generation(...) for job in jobs]\n\n# Or with asyncio\n>>> async_client = endpoint.async_client\n>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])\n\n# Pause endpoint\n>>> endpoint.pause()\n```\n\nOr if your Inference Endpoint already exists and is paused:\n\n```py\n>>> import asyncio\n>>> from huggingface_hub import get_inference_endpoint\n\n# Get endpoint + wait until initialized\n>>> endpoint = get_inference_endpoint(\"batch-endpoint\").resume().wait()\n\n# Run inference\n>>> async_client = endpoint.async_client\n>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])\n\n# Pause endpoint\n>>> endpoint.pause()\n```"}]
## 추론 실행

추론 엔드포인트가 실행되고 나면 마침내 추론을 실행할 수 있습니다!

[`InferenceEndpoint`]에는 각각 [`InferenceClient`]와 [`AsyncInferenceClient`] 객체를 반환하는 `client`와 `async_client` 두 가지 속성이 있습니다.

```py
# text_generation 태스크 실행:
>>> endpoint.client.text_generation("I am")
' not a fan of the idea of a "big-budget" movie. I think it\'s a'

# 또는 asyncio 컨텍스트에서:
>>> await endpoint.async_client.text_generation("I am")
```

추론 엔드포인트가 실행되고 있지 않으면 [`InferenceEndpointError`] 예외가 발생합니다:

```py
>>> endpoint.client
huggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.
```

[`InferenceClient`]를 사용하는 방법에 대한 자세한 내용은 [추론 가이드](../guides/inference)를 확인하세요.

## 라이프사이클 관리

이제 추론 엔드포인트를 생성하고 추론을 실행하는 방법을 배웠으므로 라이프사이클을 관리하는 방법을 살펴보겠습니다.

<Tip>

이 섹션에서는 [`~InferenceEndpoint.pause`], [`~InferenceEndpoint.resume`], [`~InferenceEndpoint.scale_to_zero`], [`~InferenceEndpoint.update`] 및 [`~InferenceEndpoint.delete`] 등의 메서드를 살펴볼 것입니다. 이들 메서드는 편의를 위해 [`InferenceEndpoint`]에 추가된 별칭입니다. 원하는 경우 `HfApi`에 정의된 일반 메서드 [`pause_inference_endpoint`], [`resume_inference_endpoint`], [`scale_to_zero_inference_endpoint`], [`update_inference_endpoint`] 및 [`delete_inference_endpoint`]를 사용할 수도 있습니다.

</Tip>

### 일시 중지 또는 0으로 스케일링

추론 엔드포인트를 사용하지 않을 때 비용을 절감하려면 [`~InferenceEndpoint.pause`]를 사용하여 일시 중지하거나 [`~InferenceEndpoint.scale_to_zero`]를 사용하여 0으로 스케일링할 수 있습니다.

<Tip>

*일시 중지* 또는 *0으로 스케일링된* 추론 엔드포인트는 비용이 들지 않습니다. 이 두 가지의 차이점은 *일시 중지된* 엔드포인트는 [`~InferenceEndpoint.resume`]를 사용하여 명시적으로 *재개*해야 한다는 점입니다. 반대로 *0으로 스케일링된* 엔드포인트는 추론 호출이 있으면 추가 콜드 스타트 지연 시간이 발생하더라도 자동으로 시작됩니다. 추론 엔드포인트는 특정 비활성 기간 후에 자동으로 0으로 스케일링되도록 구성할 수도 있습니다.

</Tip>

```py
# 엔드포인트 일시 중지 및 재개
>>> endpoint.pause()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)
>>> endpoint.resume()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
>>> endpoint.wait().client.text_generation(...)
...

# 0으로 스케일링
>>> endpoint.scale_to_zero()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
# 엔드포인트가 'running'이 아니지만 여전히 URL을 가지고 있으며 첫 번째 호출에서 다시 시작됩니다.
```

### 모델 또는 하드웨어 요구사항 업데이트

경우에 따라서는 새로운 엔드포인트를 생성하지 않고 추론 엔드포인트를 업데이트하고 싶을 수 있습니다. 호스팅된 모델 또는 모델을 실행하기 위한 하드웨어 요구사항을 업데이트할 수 있습니다. 이를 위해서는 [`~InferenceEndpoint.update`]를 사용할 수 있습니다.

```py
# 대상 모델 변경
>>> endpoint.update(repository="gpt2-large")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# 복제본 수 업데이트
>>> endpoint.update(min_replica=2, max_replica=6)
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# 더 큰 인스턴스로 업데이트
>>> endpoint.update(accelerator="cpu", instance_size="large", instance_type="c6i")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)
```

### 엔드포인트 삭제

마지막으로 추론 엔드포인트를 더 이상 사용하지 않을 경우 [`~InferenceEndpoint.delete()`]를 호출하면 됩니다.

<Tip warning={true}>

이는 취소할 수 없는 작업으로, 구성, 로그 및 사용 측정치를 포함하여 엔드포인트를 완전히 제거합니다. 삭제된 추론 엔드포인트는 복원할 수 없습니다.

</Tip>


## 종단 간 예제

추론 엔드포인트의 전형적인 사용 사례는 인프라 비용을 제한하기 위해 일괄 작업을 한 번에 처리하는 것입니다. 이 가이드에서 배운 내용을 사용하여 이 프로세스를 자동화할 수 있습니다.

```py
>>> import asyncio
>>> from huggingface_hub import create_inference_endpoint

# 엔드포인트 시작 + 초기화될 때까지 대기
>>> endpoint = create_inference_endpoint(name="batch-endpoint",...).wait()

# 추론 실행
>>> client = endpoint.client
>>> results = [client.text_generation(...) for job in jobs]  

# 또는 asyncio로
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# 엔드포인트 일시 중지
>>> endpoint.pause()
```

또는 이미 추론 엔드포인트가 존재하고 일시 중지된 경우:

```py
>>> import asyncio
>>> from huggingface_hub import get_inference_endpoint

# 엔드포인트 가져오기 + 초기화될 때까지 대기
>>> endpoint = get_inference_endpoint("batch-endpoint").resume().wait()

# 추론 실행
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# 엔드포인트 일시 중지
>>> endpoint.pause()
```
