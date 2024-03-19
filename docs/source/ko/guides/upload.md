<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 허브에 파일 업로드하기

파일과 작업을 공유하는 것은 허브의 중요한 측면입니다. 허브에 파일을 업로드하기 위한 몇 가지 옵션을 제공합니다. 이러한 기능을 독립적으로 사용하거나 라이브러리에 통합하여 사용자가 허브와 더 편리하게 상호작용할 수 있도록 할 수 있습니다. 이 가이드에서는 파일을 푸시하는 방법을 설명합니다:

- Git을 사용하지 않고
- [Git LFS](https://git-lfs.github.com/)를 사용하여 매우 큰 파일을 푸시하는 방법을 설명합니다.
- `commit` 컨텍스트 관리자를 사용합니다.
- [`~Repository.push_to_hub`] 함수를 사용합니다.

허브에 파일을 업로드할 때마다 허깅페이스 계정으로 로그인해야 합니다. 인증에 대한 자세한 내용은 [이 섹션](../quick-start#authentication)을 확인하세요.

## 파일 업로드하기

[`create_repo`]로 리파지토리를 생성했다면, [`upload_file`]을 통해 리파지토리에 파일을 업로드할 수 있습니다.

업로드할 파일의 경로, 리포지토리에서 파일을 업로드할 위치, 파일을 추가할 리포지토리의 이름을 지정합니다. 리파지토리 유형에 따라 리파지토리 유형을 `dataset`, `model`, `space`로 선택적으로 설정할 수 있습니다.


```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## 폴더 업로드

로컬 폴더를 기존 리포지토리에 업로드하려면 [`upload_folder`] 함수를 사용합니다. 업로드할 로컬 폴더의 경로를 지정합니다.
업로드할 로컬 폴더의 경로, 리포지토리에서 폴더를 업로드할 위치, 폴더를 추가할 리포지토리의 이름(
폴더를 추가할 리포지토리의 이름을 지정합니다. 리파지토리 유형에 따라 리파지토리 유형을 `데이터셋`, `모델`, `스페이스`로 선택적으로 설정할 수 있습니다.

```py
>>> huggingface_hub에서 HfApi를 가져옵니다.
>>> api = HfApi()

# 로컬 폴더의 모든 콘텐츠를 원격 스페이스로 업로드합니다.
# 기본적으로 파일은 리포지토리의 루트에 업로드됩니다.
>>> api.upload_folder(
... folder_path="/path/to/local/space",
... repo_id="username/my-cool-space",
... repo_type="space",
... )
```

기본적으로 어떤 파일을 커밋할지 여부를 알기 위해 `.gitignore` 파일이 고려된다. 기본적으로 커밋에 `.gitignore` 파일이 있는지 확인하고, 없는 경우 허브에 파일이 있는지 확인합니다. 디렉터리의 루트에 있는 `.gitignore` 파일만 사용된다는 점에 유의하세요. 하위 디렉터리에는 `.gitignore` 파일이 있는지 확인하지 않습니다.

하드코딩된 `.gitignore` 파일을 사용하지 않으려면 `allow_patterns` 및 `ignore_patterns` 인수를 사용하여 업로드할 파일을 필터링할 수 있습니다. 이 매개변수는 단일 패턴 또는 패턴 목록을 허용합니다. 패턴은 [여기](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)에 설명된 대로 표준 와일드카드(글로빙 패턴)입니다. `allow_patterns`과 `ignore_patterns`을 모두 제공하면 두 가지 제약 조건이 모두 적용됩니다.

`.gitignore` 파일과 허용/무시 패턴 외에 하위 디렉터리에 있는 모든 `.git/` 폴더는 무시됩니다.

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Upload to a specific folder
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Ignore all text logs
... )
```

`delete_patterns` 인수를 사용하여 동일한 커밋에서 리포지토리에서 삭제할 파일을 지정할 수도 있습니다.
이 방법은 파일을 푸시하기 전에 원격 폴더를 정리하고 싶은데 어떤 파일이 이미 있는지 모르는 경우
이미 존재하는지 모르는 경우에 유용합니다.

아래 예는 로컬 `./logs` 폴더를 원격 `/experiment/logs/` 폴더에 업로드하는 예입니다. txt 파일만 업로드됩니다.
하지만 그 전에는 리포지토리에 있는 모든 이전 로그가 삭제됩니다. 이 모든 것이 한 번의 커밋으로 이루어집니다.
```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Upload all local text files
...     delete_patterns="*.txt", # Delete all remote text files before
... )
```

## CLI에서 업로드

터미널에서 `huggingface-cli upload` 명령을 사용하여 허브에 파일을 직접 업로드할 수 있습니다. 내부적으로는 위에서 설명한 것과 동일한 [`upload_file`] 및 [`upload_folder`] 헬퍼를 사용합니다.

단일 파일 또는 전체 폴더를 업로드할 수 있습니다:

```bash
# Usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
>>> huggingface-cli upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> huggingface-cli upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` 및 `path_in_repo`는 선택 사항이며 암시적으로 유추할 수 있습니다. `local_path`가 설정되지 않은 경우, 이 도구는
로컬 폴더나 파일에 `repo_id`와 같은 이름이 있는지 확인합니다. 이 경우 해당 콘텐츠가 업로드됩니다.
그렇지 않으면 사용자에게 `local_path`를 명시적으로 설정하도록 요청하는 예외가 발생합니다. 어떤 경우든 `path_in_repo`가 설정되지 않으면
설정되어 있지 않으면 파일이 리포지토리의 루트에 업로드됩니다.

CLI 업로드 명령에 대한 자세한 내용은 [CLI 가이드](./cli#huggingface-cli-upload)를 참조하세요.

## 고급 기능

대부분의 경우, 허브에 파일을 업로드하는 데 [`upload_file`]과 [`upload_folder`] 이상이 필요하지 않습니다.
하지만 `huggingface_hub`에는 작업을 더 쉽게 할 수 있는 고급 기능이 있습니다. 그 기능들을 살펴봅시다!


### 차단되지 않는 업로드

메인 스레드를 차단하지 않고 데이터를 푸시하고 싶은 경우가 있습니다. 이는 특히 교육을 계속 진행하면서 로그와
아티팩트를 업로드할 때 특히 유용합니다. 이렇게 하려면 [`업로드_파일`]과 [[`업로드_폴더`] 모두에 `run_as_future` 인수를 사용할 수 있습니다.
그러면 [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
객체가 반환되며 업로드 상태를 확인하는 데 사용할 수 있습니다.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Upload in the background (non-blocking action)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Wait for the upload to complete (blocking action)
...
```

<Tip>

`run_as_future=True`를 사용하면 백그라운드 작업이 큐에 대기됩니다. 즉, 작업이 올바른 순서로 실행되도록
올바른 순서로 실행된다는 것을 의미합니다.

</Tip>

백그라운드 작업은 주로 데이터를 업로드하거나 커밋을 생성하는 데 유용하지만, 원하는 방법을 사용하여 대기열에 넣을 수 있습니다.
[`run_as_future`]. 예를 들어, 이 작업을 사용하여 리포지토리를 만든 다음 백그라운드에서 리포지토리에 데이터를 업로드할 수 있습니다. 리포지토리에
업로드 메서드에 내장된 `run_as_future` 인수는 그 주변의 별칭일 뿐입니다.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### 청크 단위로 폴더 업로드하기

[`upload_folder`]를 사용하면 전체 폴더를 허브에 쉽게 업로드할 수 있습니다. 하지만 대용량 폴더(수천 개의 파일 또는
수백 GB)의 경우 여전히 어려울 수 있습니다. 파일이 많은 폴더가 있는 경우 여러 개의 커밋에 걸쳐
업로드하는 것이 좋습니다. 업로드 중에 오류나 연결 문제가 발생하면 처음부터 다시 시작할 필요가 없습니다.
프로세스를 처음부터 다시 시작할 필요가 없습니다.

여러 커밋으로 폴더를 업로드하려면 `multi_commits=True`를 인수로 전달하면 됩니다. 내부적으로 `huggingface_hub`는
는 업로드/삭제할 파일을 나열하고 여러 커밋으로 분할합니다. "전략"(즉, 커밋을 분할하는 방법)
는 업로드할 파일의 수와 크기에 따라 결정됩니다. 모든 커밋을 푸시하기 위해 허브에 PR이 열려 있습니다. PR이
준비되면 커밋이 하나의 커밋으로 뭉쳐집니다. 완료하기 전에 프로세스가 중단된 경우 스크립트를 다시 실행하여
스크립트를 다시 실행하여 업로드를 재개할 수 있습니다. 생성된 PR이 자동으로 감지되고 업로드가 중단된 지점부터
에서 업로드가 재개됩니다. 업로드와 그 진행 상황을 더 잘 이해하려면 `multi_commits_verbose=True`를 전달하는 것이 좋습니다.
진행 상황을 더 잘 이해하려면

아래 예는 여러 커밋으로 체크포인트 폴더를 데이터셋에 업로드하는 예제입니다. 허브에 PR이 생성되고
에 PR이 생성되고 업로드가 완료되면 자동으로 병합됩니다. PR을 계속 열어두고 수동으로 검토하려면 다음과 같이 하면 됩니다.
`create_pr=True`를 전달하세요.

```py
>>> upload_folder(
...     folder_path="local/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="dataset",
...     multi_commits=True,
...     multi_commits_verbose=True,
... )
```

업로드 전략(즉, 생성되는 커밋)을 더 잘 제어하고 싶으면
저수준 [`plan_multi_commits`] 및 [`create_commits_on_pr`] 메서드를 살펴보세요.

<Tip warning={true}>

멀티 커밋`은 아직 실험적인 기능입니다. API와 동작은 향후 사전 고지 없이
예고 없이 변경될 수 있습니다.

</Tip>

### 예약된 업로드

허깅 페이스 허브를 사용하면 데이터를 쉽게 저장하고 버전업할 수 있습니다. 하지만 동일한 파일을 수천 번 업데이트할 때는 몇 가지 제한이 있습니다. 예를 들어, 배포된 Space에 대한 교육 프로세스 또는 사용자
로그를 저장하고 싶을 수 있습니다. 이러한 경우 허브에 데이터 집합으로 데이터를 업로드하는 것이 좋지만 제대로 하기가 어려울 수 있습니다. 가장 큰 이유는 데이터의 모든 업데이트를 버전으로 만들고 싶지 않기 때문인데, 그러면 git 리포지토리를 사용할 수 없게 되기 때문입니다. [`CommitScheduler`] 클래스는 이 문제에 대한 해결책을 제공합니다.

이 클래스는 로컬 폴더를 Hub에 정기적으로 푸시하는 백그라운드 작업을 실행하는 것입니다. 다음과 같이 가정해 보겠습니다.
일부 텍스트를 입력으로 받아 두 개의 번역을 생성하는 라디오 스페이스가 있다고 가정해 보겠습니다. 그런 다음 사용자가 선호하는 번역을 선택할 수 있습니다. 각 실행에 대해 입력, 출력 및 사용자 기본 설정을 저장하여 결과를 분석하려고 합니다. 이것은
[`CommitScheduler`]의 완벽한 사용 사례입니다. 허브에 데이터(잠재적으로 수백만 개의 사용자 피드백)를 저장하고 싶지만
각 사용자의 입력을 실시간으로 저장할 필요는 없습니다. 대신 데이터를 JSON 파일에 로컬로 저장한 다음
10분마다 업로드하면 됩니다. 예를 들어:

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Define the file where to save the data. Use UUID to make sure not to overwrite existing data from a previous run.
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Schedule regular uploads. Remote repo and local folder are created if they don't already exist.
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Define the function that will be called when the user submits its feedback (to be called in Gradio)
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Start Gradio
>>> with gr.Blocks() as demo:
>>>     ... # define Gradio demo + use `save_feedback`
>>> demo.launch()
```

여기까지입니다! 사용자 입력/출력 및 피드백은 허브에서 데이터 집합으로 사용할 수 있습니다. 고유한 JSON 파일 이름을 사용하면 이전 실행의 데이터나 다른 데이터의 데이터를 덮어쓰지 않도록 보장할 수 있습니다.
스페이스/복제본이 동일한 리포지토리에 동시에 푸시하는 경우.

[`CommitScheduler`]에 대한 자세한 내용은 다음과 같습니다:
- **append-only:**
    폴더에 콘텐츠만 추가한다고 가정합니다. 기존 파일에 데이터를 추가하거나 새 파일을 만들 때만
    새 파일을 만들어야 합니다. 파일을 삭제하거나 덮어쓰면 리포지토리가 손상될 수 있습니다.
- **git history**:
    스케줄러는 `every` 분마다 폴더를 커밋합니다. git 리포지토리를 너무 많이 오염시키지 않으려면
    최소값을 5분으로 설정하는 것이 좋습니다. 또한 스케줄러는 빈 커밋을 피하도록 설계되었습니다. 만약
    폴더에서 새 콘텐츠가 감지되지 않으면 예약된 커밋이 삭제됩니다.
- **errors:**
    스케줄러가 백그라운드 스레드로 실행됩니다. 클래스를 인스턴스화할 때 시작되며 절대 멈추지 않습니다. 특히
    업로드 중에 오류가 발생하면(예: 연결 문제), 스케줄러는 이를 자동으로 무시하고 다음 예약된 커밋에서
    를 다시 시도합니다.
- **thread-safety:**
    대부분의 경우 파일 잠금에 대해 걱정할 필요 없이 파일에 쓸 수 있다고 가정해도 안전합니다. 스케줄러는
    스케줄러는 업로드하는 동안 폴더에 콘텐츠를 쓰더라도 충돌하거나 손상되지 않습니다. 실제로는
    부하가 많은 앱의 경우 동시성 문제가 발생할 수 있습니다. 이 경우에는
    `scheduler.lock` 잠금을 사용하여 스레드 안전을 보장하는 것이 좋습니다. 이 잠금은 스케줄러가 폴더에서 변경 사항을 검색할 때만 차단되며
    변경 사항을 검색할 때만 잠금이 차단되며, 데이터를 업로드할 때는 차단되지 않습니다. 따라서 Space의 사용자 환경에는 영향을 미치지 않는다고 안심하셔도 됩니다.

#### 스페이스 지속성 데모

스페이스에서 허브의 데이터셋으로 데이터를 지속하는 것이 [`CommitScheduler`]의 주요 사용 사례입니다. 사용 사례에 따라
사용 사례에 따라 데이터 구조를 다르게 설정해야 할 수도 있습니다. 구조는 동시 사용자와 재시작에 대해 견고해야 하며
재시작에 견고해야 하며, 이는 종종 UUID 생성을 의미합니다. 견고성 외에도 나중에 재사용할 수 있도록 🤗 데이터 세트 라이브러리에서 읽을 수 있는 형식으로 데이터를 업로드해야 합니다. [스페이스](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
를 만들어 여러 가지 데이터 형식을 저장하는 방법을 보여줍니다(각자의 필요에 맞게 조정해야 할 수도 있습니다).

#### 사용자 지정 업로드

[`CommitScheduler`]는 데이터가 추가 전용이며 "있는 그대로" 업로드해야 한다고 가정합니다. 그러나
데이터 업로드 방식을 사용자 정의하고 싶을 수도 있습니다. [`CommitScheduler`]에서 상속하는 클래스를 생성하여 이를 수행할 수 있습니다.
에서 상속하는 클래스를 만들고 `push_to_hub` 메서드를 덮어쓰면 됩니다(원하는 방식으로 자유롭게 덮어쓰세요). 다음이 보장됩니다.
백그라운드 스레드에서 `every` 분마다 호출됩니다. 동시성 및 오류에 대해 걱정할 필요는 없지만
빈 커밋이나 중복된 데이터를 푸시하는 것과 같은 다른 측면에 주의해야 합니다.

아래의 (단순화된) 예제에서는 `push_to_hub`를 덮어쓰고 모든 PNG 파일을 단일 아카이브에 압축하여 다음과 같은 문제를 방지합니다.
허브의 리포지토리에 과부하가 걸리는 것을 방지합니다:

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. List PNG files
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # return early if nothing to commit

        # 2. Zip png files in a single archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Upload archive
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Delete local png files to avoid re-uploading them later
        for png_file in png_files:
            png_file.unlink()
```

`push_to_hub`를 덮어쓰면 [`CommitScheduler`]의 속성에 액세스할 수 있으며 특히
- [`HfApi`] 클라이언트: `api`
- 폴더 매개변수: 폴더 매개변수: `folder_path` 및 `path_in_repo`
- 리포지토리 매개변수: `repo_id`, `repo_type`, `revision`
- 스레드 잠금: `lock`

<Tip>

사용자 정의 스케줄러의 더 많은 예제는 사용 사례에 따라 다양한 구현이 포함된 [데모 스페이스](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)를 참조하세요.
사용 사례에 따라 다양한 구현이 포함되어 있습니다.

</Tip>

### create_commit

[`upload_file`] 및 [`upload_folder`] 함수는 일반적으로 사용하기 편리한 상위 수준의 API입니다. 이 함수가 익숙하지 않다면
더 낮은 수준에서 작업할 필요가 없다면 이 함수를 먼저 사용해 보세요. 그러나 커밋 수준에서 작업하고 싶다면,
[`create_commit`] 함수를 직접 사용할 수 있습니다.

[`create_commit`]이 지원하는 작업 유형은 세 가지입니다:

- 커밋 오퍼레이션 추가`]는 파일을 허브에 업로드합니다. 파일이 이미 있는 경우 파일 내용을 덮어씁니다. 이 작업은 두 개의 인수를 받습니다:

  - `path_in_repo`: 파일을 업로드할 리포지토리 경로입니다.
  - `path_or_fileobj`: 파일 시스템의 파일 경로 또는 파일과 유사한 객체. 허브에 업로드할 파일의 콘텐츠입니다.

- [`CommitOperationDelete`]는 리포지토리에서 파일 또는 폴더를 제거합니다. 이 작업은 `path_in_repo`를 인수로 받습니다.

- [`CommitOperationCopy`]는 리포지토리 내의 파일을 복사합니다. 이 작업은 세 가지 인수를 받습니다:

  - `src_path_in_repo`: 복사할 파일의 리포지토리 경로.
  - `path_in_repo`: 파일을 복사할 리포지토리 경로입니다.
  - `src_revision`: 선택 사항 - 다른 브랜치/리비전에서 파일을 복사하려는 경우 복사할 파일의 리비전입니다.

예를 들어 허브 리포지토리에서 두 개의 파일을 업로드하고 한 개의 파일을 삭제하려는 경우입니다:

1. 파일을 추가하거나 삭제하고 폴더를 삭제하려면 적절한 `CommitOperation`을 사용합니다:

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. 작업을 [`create_commit`]에 전달합니다:

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

다음 함수는 [`upload_file`] 및 [`upload_folder`] 외에도 내부적으로 [`create_commit`]을 사용합니다:

- [`delete_file`]은 허브의 리포지토리에서 단일 파일을 삭제합니다.
- [`delete_folder`]는 허브의 리포지토리에서 전체 폴더를 삭제합니다.
- [`metadata_update`]는 리포지토리의 메타데이터를 업데이트합니다.

자세한 내용은 [`HfApi`] 참조를 참조하세요.

### 커밋하기 전에 LFS 파일 미리 업로드하기

경우에 따라 커밋 호출을 하기 전에 대용량 파일을 S3에 업로드해야 할 수도 있습니다. 예를 들어 다음과 같은 경우
인메모리에 생성된 여러 개의 샤드에 있는 데이터 세트를 커밋하는 경우, 샤드를 하나씩 업로드해야 메모리 부족 문제를 피할 수 있습니다.
하나씩 업로드해야 메모리 부족 문제를 피할 수 있습니다. 해결책은 각 샤드를 리포지토리에 별도의 커밋으로 업로드하는 것입니다. 이 방법은
완벽하게 유효하지만, 이 솔루션은 수십 개의 커밋을 생성하여 잠재적으로 git 히스토리를 엉망으로 만들 수 있다는 단점이 있습니다.
이 문제를 극복하기 위해 파일을 하나씩 S3에 업로드한 다음 마지막에 하나의 커밋을 생성할 수 있습니다. 이
[`preupload_lfs_files`]와 [`create_commit`]을 함께 사용하면 가능합니다.

<Tip warning={true}>

이 방법은 고급 사용자 방법입니다. 사전 커밋의 로우 레벨 로직을 처리하는 대신 [`upload_file`], [`upload_folder`] 또는 [`create_commit`]을 직접 사용하면
를 처리하는 대신 파일을 미리 업로드하는 저수준 로직을 사용하는 것이 대부분의 경우에 적합한 방법입니다. 주요 주의 사항은
[`preupload_lfs_files`]의 주요 주의 사항은 커밋이 실제로 이루어질 때까지는 허브의 리포지토리에서 업로드 파일에 액세스할 수 없다는 것입니다.
허브의 리포지토리에 액세스할 수 없다는 것입니다. 궁금한 점이 있으면 언제든지 Discord나 GitHub 이슈로 문의해 주세요.

</Tip>

다음은 파일을 미리 업로드하는 방법을 보여주는 간단한 예시입니다:

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # List of all `CommitOperationAdd` objects that will be generated
>>> for i in range(5):
...     content = ... # generate binary content
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # Create commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

먼저, [`CommitOperationAdd`] 오브젝트를 하나씩 생성합니다. 실제 예제에서는, 여기에는
생성된 샤드를 포함합니다. 각 파일은 다음 파일을 생성하기 전에 업로드됩니다. [`preupload_lfs_files`] 단계에서는
`CommitOperationAdd` 오브젝트가 변경됩니다**. 이 객체는 [`create_commit`]에 직접 전달할 때만 사용해야 합니다. 오브젝트의 주요
오브젝트의 주요 업데이트는 **바이너리 콘텐츠가 제거**된다는 것인데, 이는 다음과 같은 경우 가비지 수집된다는 것을 의미합니다.
다른 참조를 저장하지 않으면 가비지 수집됩니다. 이미 업로드된 콘텐츠를 메모리에 보관하지 않으려는
이미 업로드된 콘텐츠를 메모리에 보관하고 싶지 않기 때문입니다. 마지막으로 모든 작업을 [`create_commit`]에 전달하여 커밋을 생성합니다. 전달할 수 있는 작업은
아직 처리되지 않은 추가 작업(추가, 삭제 또는 복사)을 전달하면 올바르게 처리됩니다.

## 대용량 업로드를 위한 팁과 요령

리포지토리에 있는 대량의 데이터를 처리할 때 주의해야 할 몇 가지 제한 사항이 있습니다. 데이터를 스트리밍하는 데 걸리는 시간을 고려하면
프로세스 마지막에 업로드/푸시가 실패하거나 hf.co에서 또는 로컬에서 작업할 때 성능 저하가 발생하는 것은 매우 성가신 일이 될 수 있습니다.

Hub에서 리포지토리를 구성하는 방법에 대한 모범 사례는 [리포지토리 제한 사항 및 권장 사항](https://huggingface.co/docs/hub/repositories-recommendations) 가이드를 참조하세요. 다음으로 업로드 프로세스를 최대한 원활하게 진행할 수 있는 몇 가지 실용적인 팁을 살펴보겠습니다.

- **작게 시작하세요**: 업로드 스크립트를 테스트할 때는 소량의 데이터로 시작하는 것이 좋습니다. 스크립트를 반복하기가 더 쉽습니다.
스크립트를 반복하는 것이 더 쉽습니다.
- **실패를 예상하세요**: 대량의 데이터를 스트리밍하는 것은 어려운 일입니다. 어떤 일이 일어날지 알 수 없지만 항상
컴퓨터, 연결, 서버 등 어떤 이유로든 한 번쯤은 실패할 수 있다는 점을 고려하는 것이 가장 좋습니다.
서버 때문이든 상관없습니다. 예를 들어, 많은 양의 파일을 업로드할 계획이라면 다음 파일을 업로드하기 전에 이미 업로드한 파일을 로컬에서 추적하는 것이 가장 좋습니다.
다음 배치를 업로드하기 전에 이미 업로드한 파일을 로컬에서 추적하는 것이 가장 좋습니다. 이미 커밋된 LFS 파일은 절대 두 번 다시 업로드되지 않습니다.
두 번 다시 업로드하지 않지만 클라이언트 측에서 확인하면 시간을 절약할 수 있습니다.
- **`hf_transfer`를 사용하세요**: 대역폭이 매우 높은 컴퓨터에서 업로드 속도를 높이기 위한 Rust 기반 [라이브러리](https://github.com/huggingface/hf_transfer)입니다.
  업로드 속도를 높이기 위한 것입니다. `hf_transfer`를 사용하려면:

    1. `huggingface_hub`를 설치할 때 `hf_transfer`를 추가로 지정합니다.
       (예: `pip install huggingface_hub[hf_transfer]`).
    2. 환경 변수로 `HF_HUB_ENABLE_HF_TRANSFER=1`을 설정합니다.

<Tip warning={true}>

`hf_transfer`는 고급 사용자 도구입니다!
테스트 및 프로덕션 준비가 완료되었습니다,
하지만 고급 오류 처리나 프록시와 같은 사용자 친화적인 기능이 부족합니다.
자세한 내용은 이 [섹션](https://huggingface.co/docs/huggingface_hub/hf_transfer)을 참조하세요.

</Tip>

## (레거시) Git LFS로 파일 업로드하기

위에서 설명한 모든 방법은 허브의 API를 사용하여 파일을 업로드합니다. 이는 허브에 파일을 업로드하는 데 권장되는 방법입니다.
하지만 로컬 리포지토리를 관리하기 위해 git 도구의 래퍼인 [`리포지토리`]도 제공합니다.

<Tip warning={true}>

리포지토리`]는 공식적으로 더 이상 사용되지 않지만, 대신 위에서 설명한 HTTP 기반 방법을 사용할 것을 권장합니다.
이 권장 사항에 대한 자세한 내용은 [이 가이드](../concepts/git_vs_http)를 참조하세요.
HTTP 기반 방식과 Git 기반 방식 간의 핵심적인 차이점을 설명합니다.

</Tip>

Git LFS는 10MB보다 큰 파일을 자동으로 처리합니다. 하지만 매우 큰 파일(5GB 이상)의 경우 Git LFS용 사용자 지정 전송 에이전트를 설치해야 합니다:

```bash
huggingface-cli lfs-enable-largefiles
```

매우 큰 파일이 있는 각 리포지토리에 대해 이 옵션을 설치해야 합니다. 설치가 완료되면 5GB보다 큰 파일을 푸시할 수 있습니다.

### 커밋 컨텍스트 관리자

`commit` 컨텍스트 관리자는 가장 일반적인 네 가지 Git 명령인 끌어오기, 추가, 커밋, 푸시를 처리합니다. `git-lfs`는 10MB보다 큰 파일을 자동으로 추적합니다. 다음 예제에서는 `commit` 컨텍스트 관리자를 사용합니다:

1. `text-files` 리포지토리에서 끌어옵니다.
2. `file.txt`에 변경 내용을 추가합니다.
3. 변경 내용을 커밋합니다.
4. 변경 내용을 `text-files` 리포지토리에 푸시합니다.

```python
>>> from huggingface_hub import Repository
>>> with Repository(local_dir="text-files", clone_from="<user>/text-files").commit(commit_message="My first file :)"):
...     with open("file.txt", "w+") as f:
...         f.write(json.dumps({"hey": 8}))
```

다음은 `commit` 컨텍스트 관리자를 사용하여 파일을 저장하고 리포지토리에 업로드하는 방법의 또 다른 예입니다:

```python
>>> import torch
>>> model = torch.nn.Transformer()
>>> with Repository("torch-model", clone_from="<user>/torch-model", token=True).commit(commit_message="My cool model :)"):
...     torch.save(model.state_dict(), "model.pt")
```

커밋을 비동기적으로 푸시하려면 `blocking=False`를 설정하세요. 커밋을 푸시하는 동안 스크립트를 계속 실행하고 싶을 때 비 블로킹 동작이 유용합니다.

```python
>>> with repo.commit(commit_message="My cool model :)", blocking=False)
```

`command_queue` 메서드로 푸시 상태를 확인할 수 있습니다:

```python
>>> last_command = repo.command_queue[-1]
>>> last_command.status
```

가능한 상태는 아래 표를 참조하세요:

| 상태      | 설명                       |
| -------- | ------------------------- |
| -1       | 푸시가 진행 중입니다.          |
| 0        | 푸시가 성공적으로 완료되었습니다. |
| Non-zero | 오류가 발생했습니다.           |

`blocking=False`인 경우, 명령이 추적되며 스크립트에서 다른 오류가 발생하더라도 모든 푸시가 완료된 경우에만 스크립트가 종료됩니다. 푸시 상태를 확인하는 데 유용한 몇 가지 추가 명령은 다음과 같습니다:

```python
# Inspect an error.
>>> last_command.stderr

# Check whether a push is completed or ongoing.
>>> last_command.is_done

# Check whether a push command has errored.
>>> last_command.failed
```

### push_to_hub

[`Repository`] 클래스에는 파일을 추가하고 커밋한 후 리포지토리로 푸시하는 [`~Repository.push_to_hub`] 함수가 있습니다. `commit` 컨텍스트 관리자와는 달리 []`~Repository.push_to_hub`]를 호출하기 전에 먼저 리포지토리에서 가져와야 합니다.

예를 들어 허브에서 리포지토리를 이미 복제했다면 로컬 디렉터리에서 `repo`를 초기화할 수 있습니다:

```python
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="path/to/local/repo")
```

로컬 클론을 [`~Repository.git_pull`]로 업데이트한 다음 파일을 Hub로 푸시합니다:

```py
>>> repo.git_pull()
>>> repo.push_to_hub(commit_message="Commit my-awesome-file to the Hub")
```

그러나 아직 파일을 푸시할 준비가 되지 않았다면 [`~Repository.git_add`] 및 [`~Repository.git_commit`]을 사용하여 파일만 추가하고 커밋할 수 있습니다:

```py
>>> repo.git_add("path/to/file")
>>> repo.git_commit(commit_message="add my first model config file :)")
```

준비가 완료되면 [`~Repository.git_push`]를 사용하여 파일을 리포지토리에 푸시합니다:

```py
>>> repo.git_push()
```
