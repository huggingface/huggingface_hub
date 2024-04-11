<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 공간 관리하기[[manage-your-space]]

이 가이드에서는 `huggingface_hub`를 사용하여 공간 런타임
([비밀 정보](https://huggingface.co/docs/hub/spaces-overview#managing-secrets),
[하드웨어](https://huggingface.co/docs/hub/spaces-gpus) 및 [저장소](https://huggingface.co/docs/hub/spaces-storage#persistent-storage))를 관리하는 방법을 살펴보겠습니다.

## 간단한 예제: 비밀 정보 및 하드웨어 구성하기.[[a-simple-example-configure-secrets-and-hardware]]

다음은 Hub에서 공간을 생성하고 설정하는 최종 예입니다.

**1. Hub에 공간 생성하기.**


```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# Gradio SDK 예제와 함께
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

**1. (bis) 공간 복제하기.**

기존의 공간에서 시작하는 대신 새로운 공간을 구축하고 싶을 때 유용할 수 있습니다.
또한 공개된 공간의 구성/설정을 제어하고 싶을 때도 유용합니다. 자세한 내용은 [`duplicate_space`]를 참조하세요.

```py
>>> api.duplicate_space("multimodalart/dreambooth-training")
```

**2. 선호하는 솔루션을 사용하여 코드 업로드하기.**

다음은 로컬 폴더 `src/`를 사용자의 컴퓨터에서 공간으로 업로드하는 예시입니다:

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

이 단계에서는 앱이 이미 무료로 Hub에서 실행 중이어야 합니다!
그러나 더 많은 비밀 정보와 업그레이드된 하드웨어로 더 구성할 수 있습니다.

**3. 비밀 정보와 변수 구성**

공간에서 작동하려면 일부 비밀 키, 토큰 또는 변수가 필요할 수 있습니다.
자세한 내용은 [문서](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)를 참조하세요.
예를 들어, Space에서 생성된 HF 토큰으로 이미지 데이터셋을 Hub에 업로드하는 경우입니다.

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

비밀 정보와 변수는 삭제할 수도 있습니다:
```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```

<Tip>
Space 내에서 비밀 정보는 환경 변수로 사용할 수 있습니다 (또는
Streamlit를 사용하는 경우 Streamlit 비밀 관리). API를 통해 가져올 필요가 없습니다!
</Tip>

<Tip warning={true}>
Space 구성 (비밀 정보 또는 하드웨어)에 대한 변경은 앱을 다시 시작합니다.
</Tip>

**보너스: Space 생성 또는 복제 시 비밀 정보와 변수 설정하기!**

Space를 생성하거나 복제할 때 비밀 정보와 변수를 설정할 수 있습니다:

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     space_variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

**4. 하드웨어 구성**

기본적으로 공간은 무료로 CPU 환경에서 실행됩니다. 하지만 하드웨어를 업그레이드하여 GPU에서 실행할 수 있습니다.
하드웨어를 업그레이드하려면 결제 카드 또는 커뮤니티 그랜트가 필요합니다. 자세한 내용은 [문서](https://huggingface.co/docs/hub/spaces-gpus)를 참조하세요.

```py
# `SpaceHardware` 열거형 사용
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# 또는 간단히 문자열 값을 전달하세요
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

하드웨어 업데이트는 즉시 이루어지지 않으며, 공간이 서버에서 다시 로드되어야 합니다.
언제든지 공간이 어떤 하드웨어에서 실행되고 있는지 확인하여 요청이 충족되었는지 확인할 수 있습니다.

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

이제 완전히 구성된 공간이 있습니다. 사용이 완료되면 공간을 다시 "cpu-classic"로 강등해야 합니다.

**보너스: 공간을 생성하거나 복제할 때 하드웨어 요청하기!**

공간이 구축되면 업그레이드된 하드웨어가 자동으로 할당됩니다.

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="cpu-upgrade",
...     space_storage="small",
...     space_sleep_time="7200", # 2 hours in secs
... )
```
```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="cpu-upgrade",
...     storage="small",
...     sleep_time="7200", # 2 hours in secs
... )
```

**5. 공간 일시 중지 및 다시 시작**

기본적으로 공간이 업그레이드된 하드웨어에서 실행 중이면 절대로 중지되지 않습니다. 그러나 요금이 부과되는 것을 피하려면 사용하지 않을 때 일시 중지하는 것이 좋습니다. 이는 [`pause_space`]를 사용하여 가능합니다. 일시 중지된 공간은 공간 소유자가 UI를 통해 또는 [`restart_space`]를 사용하여 API를 통해 다시 시작할 때까지 비활성화됩니다. 일시 중지된 모드에 대한 자세한 내용은 [이 섹션](https://huggingface.co/docs/hub/spaces-gpus#pause)을 참조하세요.

```py
# 청구를 피하기 위해 공간을 일시 중지하세요
>>> api.pause_space(repo_id=repo_id)
# (...)
# 필요할 때 다시 시작하세요
>>> api.restart_space(repo_id=repo_id)
```

다른 가능성은 공간에 대한 시간 제한을 설정하는 것입니다. 공간이 시간 제한을 초과하여 비활성화되면
공간이 슬립 상태로 전환됩니다. 공간을 방문한 방문자가 다시 시작시킵니다. [`set_space_sleep_time`]를 사용하여
시간 제한을 설정할 수 있습니다. 슬립 모드에 대한 자세한 내용은 [이 섹션](https://huggingface.co/docs/hub/spaces-gpus#sleep-time)을 참조하세요.

```py
# 비활동 후 1시간 후에 공간을 슬립 상태로 설정하세요
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

참고: 'cpu-basic' 하드웨어를 사용하는 경우 사용자 정의 슬립 시간을 구성할 수 없습니다. 공간이
48시간 동안 비활동하면 자동으로 일시 중지됩니다.

**보너스: 하드웨어를 요청하는 동안 슬립 시간 설정하기**

업그레이드된 하드웨어가 공간에 자동으로 할당됩니다.

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**보너스: 공간을 생성하거나 복제할 때 슬립 시간 설정하기!**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```
```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     hardware="t4-medium",
...     sleep_time="3600",
... )
```

**6. 공간에 지속적인 저장소 추가하기**

공간을 다시 시작할 때 지속되는 디스크 공간에 액세스할 수 있는 원하는 저장소 계층을 선택할 수 있습니다. 이는 전통적인 하드 드라이브와 같이 디스크에서 읽고 쓸 수 있음을 의미합니다. 자세한 내용은 [문서](https://huggingface.co/docs/hub/spaces-storage#persistent-storage)를 참조하세요.

```py
>>> from huggingface_hub import SpaceStorage
>>> api.request_space_storage(repo_id=repo_id, storage=SpaceStorage.LARGE)
```

또한 모든 데이터를 영구적으로 삭제하여 저장소를 삭제할 수 있습니다.
```py
>>> api.delete_space_storage(repo_id=repo_id)
```

참고: 한 번 승인된 저장소의 저장소 계층을 낮출 수 없습니다. 그렇게하려면,
먼저 저장소를 삭제한 다음 새로운 원하는 계층을 요청해야 합니다.

**보너스: 공간을 생성하거나 복제할 때 저장소 요청하기!**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_storage="large",
... )
```
```py
>>> api.duplicate_space(
...     from_id=repo_id,
...     storage="large",
... )
```

## 더 고급 기능: 공간 일시적으로 업그레이드하기![[more-advanced-temporarily-upgrade-your-space-]]

공간은 많은 다양한 사용 사례를 허용합니다. 때로는 특정 하드웨어에서 공간을 일시적으로 실행한 다음 무언가를 수행한 후 종료하고 싶을 수 있습니다.
이 섹션에서는 공간을 활용하여 필요할 때 모델을 세밀하게 조정하는 방법에 대해 탐색할 것입니다.
이는 이 특정 문제를 해결하는 한 가지 방법에 불과합니다. 이는 제안으로 받아들여져야 하며 사용 사례에 맞게 조정되어야 합니다.

모델을 세밀하게 조정하기 위한 공간이 있다고 가정해 봅시다. 입력으로 모델 ID와 데이터셋 ID를 받는 Gradio 앱입니다. 작업 흐름은 다음과 같습니다:

0. (사용자에게 모델과 데이터셋을 요청)
1. Hub에서 모델을 로드합니다.
2. Hub에서 데이터셋을 로드합니다.
3. 데이터셋으로 모델을 세밀하게 조정합니다.
4. 새 모델을 Hub에 업로드합니다.

단계 3.에서는 사용자 정의 하드웨어가 필요하지만 유료 GPU에서 공간을 항상 실행하고 싶지 않습니다.
학습을 위해 하드웨어를 동적으로 요청한 다음 종료하는 것이 해결책입니다. 하드웨어를 요청하면 공간이 다시 시작되므로 앱은 현재 수행 중인 작업을 어떻게든 "기억"해야 합니다.
이를 수행하는 여러 가지 방법이 있습니다. 이 가이드에서는 "작업 스케줄러"로서 Dataset을 사용하는 하나의 해결책을 살펴보겠습니다.

### 앱 구조[[app-skeleton]]

다음은 앱의 모습입니다. 시작할 때 예약된 작업이 있는지 확인하고 있다면
올바른 하드웨어에서 실행합니다. 작업이 완료되면 하드웨어를 무료 요금제 CPU로 다시 설정하고
사용자에게 새 작업을 요청합니다.

<Tip warning={true}>
일반적인 데모처럼 병렬 액세스를 지원하지 않습니다.
특히, 학습이 진행되는 동안 인터페이스가 비활성화됩니다.
리포지토리를 개인으로 설정하여 단일 사용자임을 보장하는 것이 좋습니다.
</Tip>

```py
# 공간은 하드웨어를 요청하기 위해 토큰이 필요합니다: 비밀로 설정하세요!
HF_TOKEN = os.environ.get("HF_TOKEN")

# 공간 소유 리포지토리 ID
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# 공간 시작 시 예약된 작업을 확인합니다. 예약된 작업이 있는 경우 모델을 세밀하게 조정합니다. 그렇지 않은 경우,
# 새 작업을 요청할 수 있는 인터페이스를 표시합니다.
task = get_task()
if task is None:
    # Gradio 앱 시작
    def gradio_fn(task):
        # 사용자 요청 시 작업 추가 및 하드웨어 요청
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # GPU를 사용 중인지 확인합니다.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # 그렇다면, 기본 모델을 데이터셋으로 세밀하게 조정합니다!
        train_and_upload(task)

        # 그런 다음, 작업을 "완료"로 표시합니다.
        mark_as_done(task)

        # 잊지 말아야 할 것: CPU 하드웨어로 다시 설정
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### 작업 스케줄러[[task-scheduler]]

작업 스케줄링은 여러 가지 방법으로 수행할 수 있습니다. 여기에는 간단한 CSV를 데이터셋으로 사용하여 작업 스케줄링을 하는 예시가 있습니다.

```py
# 'tasks.csv' 파일을 포함하는 데이터셋의 데이터셋 ID.
# 여기에는 입력 (기본 모델 및 데이터셋)과 상태 (보류 중 또는 완료)가 포함된 'tasks.csv'의 기본 예제가 있습니다.
#     multimodalart/sd-fine-tunable,Wauplin/concept-1,DONE
#     multimodalart/sd-fine-tunable,Wauplin/concept-2,PENDING
TASK_DATASET_ID = "Wauplin/dreambooth-task-scheduler"

def _get_csv_file():
    return hf_hub_download(repo_id=TASK_DATASET_ID, filename="tasks.csv", repo_type="dataset", token=HF_TOKEN)

def get_task():
    with open(_get_csv_file()) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] == "PENDING":
                return row[0], row[1] # model_id, dataset_id

def add_task(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # 작업 추가의 빠르고 더러운 방법
        path_or_fileobj=(tasks + f"\n{model_id},{dataset_id},PENDING").encode()
    )

def mark_as_done(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # 작업을 완료로 설정하는 빠르고 더러운 방법
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```
