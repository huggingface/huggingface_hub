<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 웹훅 서버[[webhooks-server]]

웹훅은 MLOps 관련 기능의 기반이 됩니다. 이를 통해 특정 저장소의 새로운 변경 사항을 수신하거나,
관심 있는 특정 사용자/조직에 속한 모든 저장소의 변경 사항을 받아볼 수 있습니다.
이 가이드에서는 `huggingface_hub`를 활용하여 웹훅을 수신하는 서버를 만들고 Space에 배포하는 방법을 설명합니다. 
이를 위해서는 Huggingface Hub의 웹훅 개념에 대해 익숙해야 합니다. 
웹훅 자체에 대해 더 자세히 알아보려면 이 [가이드](https://huggingface.co/docs/hub/webhooks)를 먼저 읽어보세요.  

이 가이드에서 사용할 기본 클래스는 [`WebhooksServer`]입니다. 
이 클래스는 Huggingface Hub에서 웹훅을 받을 수 있는 서버를 쉽게 구성할 수 있습니다. 서버는 [Gradio](https://gradio.app/) 앱을 기반으로 합니다. 
이 서버에는 사용자를 위한 지침을 표시하는 UI와 웹훅을 수신하는 API가 있습니다.

> [!TIP]
> 웹훅 서버의 실행 예시를 보려면 [Spaces CI Bot](https://huggingface.co/spaces/spaces-ci-bot/webhook)을 확인하세요. 
> 이것은 Space의 PR이 열릴 때마다 임시 환경을 실행하는 Space입니다.

> [!WARNING]
> 이것은 [실험적 기능](../package_reference/environment_variables#hfhubdisableexperimentalwarning)입니다. 
> 본 API는 현재 개선 작업 중이며, 향후 사전 통지 없이 주요 변경 사항이 도입될 수 있습니다. 
> requirements에서 `huggingface_hub`의 버전을 고정하는 것을 권장합니다.


## 엔드포인트 생성[[create-an-endpoint]]

웹훅 엔드포인트를 구현하는 것은 함수에 데코레이터를 추가하는 것만큼 간단합니다. 
주요 개념을 설명하기 위해 첫 번째 예시를 살펴보겠습니다:

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # 데이터 세트가 업데이트되면 학습 작업을 트리거합니다.
        ...
```

이 코드 스니펫을 `'app.py'`라는 파일에 저장하고 `'python app.py'`로 실행하면 다음과 같은 메시지가 표시될 것입니다:

```text
Webhook secret is not defined. This means your webhook endpoints will be open to everyone.
To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization:
        `app = WebhooksServer(webhook_secret='my_secret', ...)`
For more details about webhook secrets, please refer to https://huggingface.co/docs/hub/webhooks#webhook-secret.
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://1fadb0f52d8bf825fc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces

Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
Go to https://huggingface.co/settings/webhooks to setup your webhooks.
```

축하합니다! 웹훅 서버를 실행했습니다! 정확히 어떤 일이 일어났는지 살펴보겠습니다:

1. [`webhook_endpoint`]로 함수에 데코레이터를 추가하면 백그라운드에서 [`WebhooksServer`] 객체가 생성됩니다. 
볼 수 있듯이 이 서버는 http://127.0.0.1:7860 에서 실행되는 Gradio 앱입니다. 
이 URL을 브라우저에서 열면 등록된 웹훅에 대한 지침이 있는 랜딩 페이지를 볼 수 있습니다.
2. Gradio 앱은 내부적으로 FastAPI 서버입니다. 새로운 POST 경로 `/webhooks/trigger_training`이 추가되었습니다. 
이 경로는 웹훅을 수신하고 트리거될 때 `trigger_training` 함수를 실행합니다. 
FastAPI는 자동으로 페이로드를 구문 분석하고 [`WebhookPayload`] 객체로 함수에 전달합니다. 
이 `pydantic` 객체에는 웹훅을 트리거한 이벤트에 대한 모든 정보가 포함되어 있습니다.
3. Gradio 앱은 인터넷에서 요청을 받을 수 있는 터널도 열었습니다. 
이것은 흥미로운 부분으로, https://huggingface.co/settings/webhooks 에서 로컬 머신을 가리키는 웹훅을 구성할 수 있습니다. 
이를 통해 웹훅 서버를 디버깅하고 Space에 배포하기 전에 빠르게 반복할 수 있습니다.
4. 마지막으로 로그에는 서버가 현재 비밀로 보호되지 않는다고 알려줍니다. 
이것은 로컬 디버깅에는 문제가 되지 않지만 나중에 고려해야 할 사항입니다.

> [!WARNING]
> 기본적으로 서버는 스크립트 끝에서 시작됩니다. 
> 주피터 노트북에서 실행 중이라면 `decorated_function.run()`을 호출하여 서버를 수동으로 시작할 수 있습니다. 
> 고유한 서버를 사용하기 때문에 여러 엔드포인트가 있더라도 서버를 한 번만 시작하면 됩니다.


## 웹훅 설정하기[[configure-a-webhook]]

웹훅 서버를 실행하고 있으므로, 이제 메시지를 수신하기 위해 웹훅을 구성해야 합니다.
https://huggingface.co/settings/webhooks 로 이동하여 "Add a new webhook"을 클릭하고 웹훅을 구성하세요. 
모니터링할 대상 저장소와 웹훅 URL `https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training`을 설정하세요.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/configure_webhook.png"/>
</div>

이걸로 끝입니다! 이제 대상 저장소를 업데이트하면 웹훅을 트리거할 수 있습니다. 예를 들면, 커밋 푸시가 그 방법이 될 수 있습니다.
웹훅의 Activity 탭에서 트리거된 이벤트를 확인할 수 있습니다. 이제 작동하는 구성이 있으므로 테스트하고 빠르게 반복할 수 있습니다. 
코드를 수정하고 서버를 다시 시작하면 공개 URL이 변경될 수 있습니다. 
필요한 경우 Hub에서 웹훅 구성을 업데이트하세요.

## Space에 배포하기[[deploy-to-a-space]]

이제 작동하는 웹훅 서버가 마련되었으므로, 다음 목표는 이를 Space에 배포하는 것입니다. https://huggingface.co/new-space 에 가서 Space를 생성합니다. 
이름을 지정하고, Gradio SDK를 선택한 다음 "Create Space"를 클릭합니다. 코드를 `app.py` 파일로 Space에 업로드합니다.
Space가 자동으로 시작됩니다!
Space에 대한 자세한 내용은 이 [가이드](https://huggingface.co/docs/hub/spaces-overview)를 참조하세요.

웹훅 서버가 이제 공개 Space에서 실행 중입니다. 대부분의 경우 비밀번호로 보안을 설정하고 싶을 것입니다.
Space 설정 > "Repository secrets" 섹션 > "Add a secret" 로 이동합니다. `WEBHOOK_SECRET` 환경 변수에 원하는 값을 설정합니다. 
[Webhooks 설정](https://huggingface.co/settings/webhooks)으로 돌아가서 웹훅 구성에 비밀번호를 설정합니다. 
이제 올바른 비밀번호가 있는 요청만 서버에서 허용됩니다.

이게 전부입니다! Space가 이제 Hub의 웹훅을 수신할 준비가 되었습니다.
무료 하드웨어인 'cpu-basic'에서 Space를 실행 시, 48시간 동안 비활성화되면 종료된다는 점을 유념하세요. 
영구적인 Space가 필요한 경우 [업그레이드된 하드웨어](https://huggingface.co/docs/hub/spaces-gpus#hardware-specs)를 설정해야 합니다.

## 고급 사용법[[advanced-usage]]

위의 가이드에서는 [`WebhooksServer`]를 설정하는 가장 빠른 방법에 대해 설명했습니다. 
이 섹션에서는 이를 더욱 사용자 정의하는 방법을 살펴보겠습니다.

### 다중 엔드포인트[[multiple-endpoints]]

동일한 서버에 여러 엔드포인트를 등록할 수 있습니다. 
예를 들어, 하나의 엔드포인트는 학습 작업을 트리거하고 다른 엔드포인트는 모델 평가를 트리거하도록 할 수 있습니다. 
이를 위해 여러 개의 `@webhook_endpoint` 데코레이터를 추가하면 됩니다:

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # 데이터 세트가 업데이트되면 학습 작업을 트리거합니다.
        ...

@webhook_endpoint
async def trigger_evaluation(payload: WebhookPayload) -> None:
    if payload.repo.type == "model" and payload.event.action == "update":
        # 모델이 업데이트되면 평가 작업을 트리거합니다. 
        ...
```

이렇게 하면 두 개의 엔드포인트가 생성됩니다:

```text
(...)
Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_evaluation
```

### 사용자 정의 서버[[custom-server]]

더 많은 유연성을 얻기 위해 [`WebhooksServer`] 객체를 직접 생성할 수도 있습니다. 
이것은 서버의 랜딩 페이지를 사용자 정의하고자 할 때 유용합니다. 
기본 페이지를 덮어쓸 [Gradio UI](https://gradio.app/docs/#blocks)를 전달하여 이를 수행할 수 있습니다. 
예를 들어, 사용자를 위한 지침을 추가하거나 웹훅을 수동으로 트리거하는 양식을 추가할 수 있습니다. 
[`WebhooksServer`]를 생성할 때, [`~WebhooksServer.add_webhook`] 데코레이터를 사용하여 새로운 웹훅을 등록할 수 있습니다.

전체 예제는 다음과 같습니다:

```python
import gradio as gr
from fastapi import Request
from huggingface_hub import WebhooksServer, WebhookPayload

# 1. UI 정의
with gr.Blocks() as ui:
    ...

# 2. 사용자 정의 UI와 시크릿으로 WebhooksServer 생성
app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

# 3. 명시적 이름으로 웹훅 등록
@app.add_webhook("/say_hello")
async def hello(payload: WebhookPayload):
    return {"message": "hello"}

# 4. 암시적 이름으로 웹훅 등록
@app.add_webhook
async def goodbye(payload: WebhookPayload):
    return {"message": "goodbye"}

# 5. 서버 시작 (선택 사항)
app.run()
```

1. Gradio 블록을 사용하여 사용자 정의 UI를 정의합니다. 이 UI는 서버의 랜딩 페이지에 표시됩니다.
2. 사용자 정의 UI와 시크릿으로 [`WebhooksServer`] 객체를 생성합니다. 
시크릿은 선택 사항이며 `WEBHOOK_SECRET` 환경 변수로 설정할 수 있습니다. 
3. 명시적 이름으로 웹훅을 등록합니다. 이렇게 하면 `/webhooks/say_hello` 엔드포인트가 생성됩니다.
4. 암시적 이름으로 웹훅을 등록합니다. 이렇게 하면 `/webhooks/goodbye` 엔드포인트가 생성됩니다.
5. 서버를 시작합니다. 이것은 선택 사항이며 스크립트 끝에서 자동으로 서버가 시작됩니다.
