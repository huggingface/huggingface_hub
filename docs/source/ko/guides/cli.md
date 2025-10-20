<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 명령줄 인터페이스 (CLI) [[command-line-interface]]

`huggingface_hub` Python 패키지는 `hf`라는 내장 CLI를 함께 제공합니다. 이 도구를 사용하면 터미널에서 Hugging Face Hub와 직접 상호 작용할 수 있습니다. 계정에 로그인하고, 리포지토리를 생성하고, 파일을 업로드 및 다운로드하는 등의 다양한 작업을 수행할 수 있습니다. 또한 머신을 구성하거나 캐시를 관리하는 데 유용한 기능도 제공합니다. 이 가이드는 CLI의 주요 기능과 사용 방법에 관해 설명합니다.

## 시작하기 [[getting-started]]

먼저, CLI를 설치해 보세요:

```
>>> pip install -U "huggingface_hub"
```

> [!TIP]
> CLI는 기본 `huggingface_hub` 패키지에 포함되어 있습니다.

설치가 완료되면, CLI가 올바르게 설정되었는지 확인할 수 있습니다:

```
>>> hf --help
usage: hf <command> [<args>]

positional arguments:
  {auth,cache,download,repo,repo-files,upload,upload-large-folder,env,version,lfs-enable-largefiles,lfs-multipart-upload}
                        hf command helpers
    auth                Manage authentication (login, logout, etc.).
    cache               Manage local cache directory.
    download            Download files from the Hub
    repo                Manage repos on the Hub.
    repo-files          Manage files in a repo on the Hub.
    upload              Upload a file or a folder to the Hub. Recommended for single-commit uploads.
    upload-large-folder
                        Upload a large folder to the Hub. Recommended for resumable uploads.
    env                 Print information about the environment.
    version             Print information about the hf version.

options:
  -h, --help            show this help message and exit
```

CLI가 제대로 설치되었다면 CLI에서 사용 가능한 모든 옵션 목록이 출력됩니다. `command not found: hf`와 같은 오류 메시지가 표시된다면 [설치](../installation) 가이드를 확인하세요.

> [!TIP]
> `--help` 옵션을 사용하면 명령어에 대한 자세한 정보를 얻을 수 있습니다. 언제든지 사용 가능한 모든 옵션과 그 세부 사항을 확인할 수 있습니다. 예를 들어 `hf upload --help`는 CLI를 사용하여 파일을 업로드하는 구체적인 방법을 알려줍니다.

### 다른 방법으로 설치하기 [[alternative-install]]

#### uv 사용하기 [[using-uv]]

[uv](https://docs.astral.sh/uv/)를 사용하면 `hf` CLI를 설치하거나, 설치 없이 바로 실행할 수 있습니다. 먼저 uv를 설치하세요 (PATH에 `uv`와 `uvx`가 추가됩니다):

```bash
>>> curl -LsSf https://astral.sh/uv/install.sh | sh
```

영구적으로 도구를 설치해 어디에서나 사용하려면:

```bash
>>> uv tool install "huggingface_hub"
>>> hf --help
```

전역 설치 없이 일회성으로 실행하려면 `uvx`를 사용하세요:

```bash
>>> uvx --from huggingface_hub hf --help
```

#### Homebrew 사용하기 [[using-homebrew]]

[Homebrew](https://brew.sh/)를 사용하여 CLI를 설치할 수도 있습니다:

```bash
>>> brew install huggingface-cli
```

Homebrew huggingface에 대한 자세한 내용은 [여기](https://formulae.brew.sh/formula/huggingface-cli)에서 확인할 수 있습니다.

## hf auth login [[hf-login]]

Hugging Face Hub에 접근하는 대부분의 작업(비공개 리포지토리 액세스, 파일 업로드, PR 제출 등)을 위해서는 Hugging Face 계정에 로그인해야 합니다. 로그인을 하기 위해서 [설정 페이지](https://huggingface.co/settings/tokens)에서 생성한 [사용자 액세스 토큰](https://huggingface.co/docs/hub/security-tokens)이 필요하며, 이 토큰은 Hub에서의 사용자 인증에 사용됩니다. 파일 업로드나 콘텐츠 수정을 위해선 쓰기 권한이 있는 토큰이 필요합니다.
토큰을 받은 후에 터미널에서 다음 명령을 실행하세요:

```bash
>>> hf auth login
```

이 명령은 토큰을 입력하라는 메시지를 표시합니다. 토큰을 복사하여 붙여넣고 Enter 키를 입력합니다. 그런 다음 토큰을 git 자격 증명으로 저장할지 묻는 메시지가 표시됩니다. 로컬에서 `git`을 사용할 계획이라면 Enter 키를 입력합니다(기본값은 yes). 마지막으로 Hub에서 토큰의 유효성을 검증한 후 로컬에 저장합니다.

```
_|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
_|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
_|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
_|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
_|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token:
Add token as git credential? (Y/n)
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

프롬프트를 거치지 않고 바로 로그인하고 싶다면, 명령줄에서 토큰을 직접 입력할 수도 있습니다. 하지만 보안을 더욱 강화하기 위해서는 명령 기록에 토큰을 남기지 않고, 환경 변수를 통해 토큰을 전달하는 방법이 바람직합니다.

```bash
# Or using an environment variable
>>> hf auth login --token $HUGGINGFACE_TOKEN --add-to-git-credential
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

[이 단락](../quick-start#authentication)에서 인증에 대한 더 자세한 내용을 확인할 수 있습니다.

## hf auth whoami [[hf-whoami]]

로그인 여부를 확인하기 위해 `hf auth whoami` 명령어를 사용할 수 있습니다. 이 명령어는 옵션이 없으며, 간단하게 사용자 이름과 소속된 조직들을 출력합니다:

```bash
hf auth whoami
Wauplin
orgs:  huggingface,eu-test,OAuthTesters,hf-accelerate,HFSmolCluster
```

로그인하지 않은 경우 오류 메시지가 출력됩니다.

## hf auth logout [[hf-auth-logout]]

이 명령어를 사용하여 로그아웃할 수 있습니다. 실제로는 컴퓨터에 저장된 토큰을 삭제합니다.

하지만 `HF_TOKEN` 환경 변수를 사용하여 로그인했다면, 이 명령어로는 로그아웃할 수 없습니다([참조]((../package_reference/environment_variables#hftoken))). 대신 컴퓨터의 환경 설정에서 `HF_TOKEN` 변수를 제거하면 됩니다.

## hf download [[hf-download]]


`hf download` 명령어를 사용하여 Hub에서 직접 파일을 다운로드할 수 있습니다. [다운로드](./download) 가이드에서 설명된 [`hf_hub_download`], [`snapshot_download`] 헬퍼 함수를 사용하여 반환된 경로를 터미널에 출력합니다. 우리는 아래 예시에서 가장 일반적인 사용 사례를 살펴볼 것입니다. 사용 가능한 모든 옵션을 보려면 아래 명령어를 실행해보세요:

```bash
hf download --help
```

### 파일 한 개 다운로드하기 [[download-a-single-file]]

리포지토리에서 파일 하나를 다운로드하고 싶다면, repo_id와 다운받고 싶은 파일명을 아래와 같이 입력하세요:

```bash
>>> hf download gpt2 config.json
downloading https://huggingface.co/gpt2/resolve/main/config.json to /home/wauplin/.cache/huggingface/hub/tmpwrq8dm5o
(…)ingface.co/gpt2/resolve/main/config.json: 100%|██████████████████████████████████| 665/665 [00:00<00:00, 2.49MB/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

이 명령어를 실행하면 항상 마지막 줄에 파일 경로를 출력합니다.

### 전체 리포지토리 다운로드하기 [[download-an-entire-repository]]

리포지토리의 모든 파일을 다운로드하고 싶을 때에는 repo id만 입력하면 됩니다:

```bash
>>> hf download HuggingFaceH4/zephyr-7b-beta
Fetching 23 files:   0%|                                                | 0/23 [00:00<?, ?it/s]
...
...
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### 여러 파일 다운로드하기 [[download-multiple-files]]

리포지토리의 전체 폴더를 다운로드하지 않고 한 번에 여러 파일을 다운로드할 수도 있습니다. 이를 위한 두 가지 방법이 있습니다. 다운로드하고자 하는 파일들의 목록이 정해져 있다면, 해당 파일명을 순서대로 입력하면 됩니다:

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files:   0%|                                                                        | 0/2 [00:00<?, ?it/s]
downloading https://huggingface.co/gpt2/resolve/11c5a3d5811f50298f278a704980280950aedb10/model.safetensors to /home/wauplin/.cache/huggingface/hub/tmpdachpl3o
(…)8f278a7049802950aedb10/model.safetensors: 100%|██████████████████████████████| 8.09k/8.09k [00:00<00:00, 40.5MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.76it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

또 다른 방법은 `--include`와 `--exclude` 옵션을 사용하여 원하는 파일을 필터링하는 것입니다. 예를 들어, [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)의 모든 safetensors 파일을 다운로드하되 FP16 정밀도의 파일은 제외하고 싶다면 다음과 같이 실행할 수 있습니다:

```bash
>>> hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*
Fetching 8 files:   0%|                                                                         | 0/8 [00:00<?, ?it/s]
...
...
Fetching 8 files: 100%|█████████████████████████████████████████████████████████████████████████| 8/8 (...)
/home/wauplin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
```

### 데이터 세트 또는 Space 다운로드하기 [[download-a-dataset-or-a-space]]

앞서 소개된 예시들을 통해 모델 리포지토리에서 다운로드하는 방법을 배웠습니다. 데이터 세트나 Space를 다운로드하고자 할 때는 `--repo-type` 옵션을 사용하세요:

```bash
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
>>> hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat
>>> hf download HuggingFaceH4/zephyr-chat --repo-type space

...
```

### 특정 리비전 다운로드하기 [[download-a-specific-revision]]

따로 리비전을 지정하지 않는다면 기본적으로 main 브랜치의 최신 커밋에서 파일을 다운로드합니다. 특정 리비전(커밋 해시, 브랜치 이름 또는 태그)에서 다운로드하려면 `--revision` 옵션을 사용하세요:

```bash
>>> hf download bigcode/the-stack --repo-type dataset --revision v1.1
...
```

### 로컬 폴더에 다운로드하기 [[download-to-a-local-folder]]

Hub에서 파일을 다운로드하는 권장되고 기본적인 방법은 캐시 시스템을 사용하는 것입니다. 그러나 특정한 경우에는 파일을 지정된 폴더로 다운로드하고 옮기고 싶을 수 있습니다. 이는 git 명령어와 유사한 워크플로우를 만드는데 도움이 됩니다. `--local_dir` 옵션을 사용하여 이 작업을 수행할 수 있습니다.

> [!WARNING]
> 로컬 폴더에 다운로드하는 것에는 몇 가지 단점이 있습니다. `--local-dir` 명령어를 사용하기 전에 [다운로드](./download#download-files-to-local-folder) 가이드에서 해당 내용을 확인해보세요.

```bash
>>> hf download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir .
...
./model-00001-of-00002.safetensors
```

### 캐시 디렉터리 지정하기 [[specify-cache-directory]]

기본적으로 모든 파일은 `HF_HOME` [환경 변수](../package_reference/environment_variables#hfhome)에서 정의한 캐시 디렉터리에 다운로드됩니다. `--cache-dir`을 사용하여 직접 캐시 위치를 지정할 수 있습니다:

```bash
>>> hf download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### 토큰 설정하기 [[specify-a-token]]

비공개 또는 접근이 제한된 리포지토리들에 접근하기 위해서는 토큰이 필요합니다. 기본적으로 로컬에 저장된 토큰(`hf auth login`)이 사용됩니다. 직접 인증하고 싶다면 `--token` 옵션을 사용해보세요:

```bash
>>> hf download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### 조용한 모드 [[quiet-mode]]

`hf download` 명령은 상세한 정보를 출력합니다. 경고 메시지, 다운로드된 파일 정보, 진행률 등이 포함됩니다. 이 모든 출력을 숨기려면 `--quiet` 옵션을 사용하세요. 이 옵션을 사용하면 다운로드된 파일의 경로가 표시되는 마지막 줄만 출력됩니다. 이 기능은 스크립트에서 다른 명령어로 출력을 전달하고자 할 때 유용할 수 있습니다.

```bash
>>> hf download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

## hf upload [[hf-upload]]

`hf upload` 명령어로 Hub에 직접 파일을 업로드할 수 있습니다. [업로드](./upload) 가이드에서 설명된 [`upload_file`], [`upload_folder`] 헬퍼 함수를 사용합니다. 우리는 아래 예시에서 가장 일반적인 사용 사례를 살펴볼 것입니다. 사용 가능한 모든 옵션을 보려면 아래 명령어를 실행해보세요:

```bash
>>> hf upload --help
```

### 전체 폴더 업로드하기 [[upload-an-entire-folder]]

이 명령어의 기본 사용법은 다음과 같습니다:

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
```

현재 디텍터리를 리포지토리의 루트 위치에 업로드하려면, 아래 명령어를 사용하세요:

```bash
>>> hf upload my-cool-model . .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

> [!TIP]
> 리포지토리가 아직 존재하지 않으면 자동으로 생성됩니다.

또한, 특정 폴더만 업로드하는 것도 가능합니다:

```bash
>>> hf upload my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

마지막으로, 리포지토리의 특정 위치에 폴더를 업로드할 수 있습니다:

```bash
>>> hf upload my-cool-model ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/my-cool-model/tree/main/data/train
```

### 파일 한 개 업로드하기 [[upload-a-single-file]]

컴퓨터에 있는 파일을 가리키도록 `local_path`를 설정함으로써 파일 한 개를 업로드할 수 있습니다. 이때, `path_in_repo`는 선택사항이며 로컬 파일 이름을 기본값으로 사용합니다:

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors
```

파일 한 개를 특정 디렉터리에 업로드하고 싶다면, `path_in_repo`를 그에 맞게 설정하세요:

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/vae/model.safetensors
```

### 여러 파일 업로드하기 [[upload-multiple-files]]

전체 폴더를 업로드하지 않고 한 번에 여러 파일을 업로드하려면 `--include`와 `--exclude` 옵션을 사용해보세요. 리포지토리에 있는 파일을 삭제하면서 새 파일을 업로드하는 `--delete` 옵션과 함께 사용할 수 있습니다. 아래 예시는 `/logs` 안의 파일을 제외한 모든 파일을 업로드하고 원격 파일들을 삭제함으로써 로컬 Space를 동기화하는 방법을 보여줍니다:

```bash
# Sync local Space with Hub (upload new files except from logs/, delete removed files)
>>> hf upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
...
```

### 데이터 세트 또는 Space에 업로드하기 [[upload-to-a-dataset-or-space]]

데이터 세트나 Space에 업로드하려면 `--repo-type` 옵션을 사용하세요:

```bash
>>> hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
...
```

### 조직에 업로드하기 [[upload-to-an-organization]]

개인 리포지토리 대신 조직이 소유한 리포지토리에 파일을 업로드하려면 `repo_id`를 입력해야 합니다:

```bash
>>> hf upload MyCoolOrganization/my-cool-model . .
https://huggingface.co/MyCoolOrganization/my-cool-model/tree/main/
```

### 특정 개정에 업로드하기 [[upload-to-a-specific-revision]]

기본적으로 파일은 `main` 브랜치에 업로드됩니다. 다른 브랜치나 참조에 파일을 업로드하려면 `--revision` 옵션을 사용하세요:

```bash
# Upload files to a PR
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**참고:** `revision`이 존재하지 않고 `--create-pr` 옵션이 설정되지 않은 경우, `main` 브랜치에서 자동으로 새 브랜치가 생성됩니다.

### 업로드 및 PR 생성하기 [[upload-and-create-a-pr]]

리포지토리에 푸시할 권한이 없다면, PR을 생성하여 작성자들에게 변경하고자 하는 내용을 알려야 합니다. 이를 위해서 `--create-pr` 옵션을 사용할 수 있습니다:

```bash
# Create a PR and upload the files to it
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### 정기적으로 업로드하기 [[upload-at-regular-intervals]]

리포지토리에 정기적으로 업데이트하고 싶을 때, `--every` 옵션을 사용할 수 있습니다. 예를 들어, 모델을 훈련하는 중에 로그 폴더를 10분마다 업로드하고 싶다면 다음과 같이 사용할 수 있습니다:

```bash
# Upload new logs every 10 minutes
hf upload training-model logs/ --every=10
```

### 커밋 메시지 지정하기 [[specify-a-commit-message]]

`--commit-message`와 `--commit-description`을 사용하여 기본 메시지 대신 사용자 지정 메시지와 설명을 커밋에 설정하세요:

```bash
>>> hf upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### 토큰 지정하기 [[specify-a-token]]

파일을 업로드하려면 토큰이 필요합니다. 기본적으로 로컬에 저장된 토큰(`hf auth login`)이 사용됩니다. 직접 인증하고 싶다면 `--token` 옵션을 사용해보세요:

```bash
>>> hf upload Wauplin/my-cool-model ./models . --token=hf_****
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### 조용한 모드 [[quiet-mode]]

기본적으로 `hf upload` 명령은 상세한 정보를 출력합니다. 경고 메시지, 업로드된 파일 정보, 진행률 등이 포함됩니다. 이 모든 출력을 숨기려면 `--quiet` 옵션을 사용하세요. 이 옵션을 사용하면 업로드된 파일의 URL이 표시되는 마지막 줄만 출력됩니다. 이 기능은 스크립트에서 다른 명령어로 출력을 전달하고자 할 때 유용할 수 있습니다.

```bash
>>> hf upload Wauplin/my-cool-model ./models . --quiet
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

## hf cache ls [[hf-cache-ls]]

로컬 캐시에 어떤 리포지토리나 수정 버전이 저장되어 있는지 확인하려면 `hf cache ls`를 사용하세요. 기본 출력은 리포지토리 단위 요약입니다.

```bash
>>> hf cache ls
ID                                   SIZE   LAST_ACCESSED LAST_MODIFIED REFS
------------------------------------ ------- ------------- ------------- -------------------
dataset/glue                         116.3K 4 days ago     4 days ago     2.4.0 main 1.17.0
dataset/google/fleurs                 64.9M 1 week ago     1 week ago     main refs/pr/1
model/Jean-Baptiste/camembert-ner    441.0M 2 weeks ago    16 hours ago   main
model/bert-base-cased                  1.9G 1 week ago     2 years ago
model/t5-base                          10.1K 3 months ago   3 months ago   main
model/t5-small                        970.7M 3 days ago     3 days ago     main refs/pr/1

Found 6 repo(s) for a total of 12 revision(s) and 3.4G on disk.
```

`--revisions` 옵션과 `--filter` 표현식을 조합하면 특정 스냅샷만 추려 볼 수 있습니다.

```bash
>>> hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
ID                                   REVISION            SIZE   LAST_MODIFIED REFS
------------------------------------ ------------------ ------- ------------- -------------------
model/bert-base-cased                6d1d7a1a2a6cf4c2    1.9G  2 years ago
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main

Found 2 repo(s) for a total of 2 revision(s) and 3.0G on disk.
```

`--format json`, `--format csv`, `--quiet`, `--cache-dir` 등 다양한 옵션으로 출력 형식을 조정할 수 있습니다. 자세한 내용은 [캐시 관리](./manage-cache#scan-your-cache) 가이드를 참고하세요.

`hf cache ls --quiet`로 추린 식별자를 `hf cache rm`에 바로 파이프하면 오래된 항목을 한 번에 정리할 수 있습니다.

```bash
>>> hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

## hf cache rm [[hf-cache-rm]]

캐시에서 특정 리포지토리나 수정 버전을 삭제하려면 `hf cache rm`을 사용합니다. 리포지토리 식별자나 수정 버전 해시를 하나 이상 전달하면 됩니다. `--dry-run`으로 미리보기, `--yes`로 확인창 건너뛰기, `--cache-dir`로 다른 경로 지정이 가능합니다.

## hf cache prune [[hf-cache-prune]]

참조되지 않는(detached) 수정 버전만 한꺼번에 제거하려면 `hf cache prune`을 실행하세요. `--dry-run`, `--yes`, `--cache-dir` 옵션 역시 동일하게 사용할 수 있습니다.

## hf env [[hf-env]]

`hf env` 명령어는 사용자의 컴퓨터 설정에 대한 상세한 정보를 보여줍니다. 이는 [GitHub](https://github.com/huggingface/huggingface_hub)에서 문제를 제출할 때, 관리자가 문제를 파악하고 해결하는 데 도움이 됩니다.

```bash
>>> hf env

Copy-and-paste the text below in your GitHub issue.

- huggingface_hub version: 0.19.0.dev0
- Platform: Linux-6.2.0-36-generic-x86_64-with-glibc2.35
- Python version: 3.10.12
- Running in iPython ?: No
- Running in notebook ?: No
- Running in Google Colab ?: No
- Token path ?: /home/wauplin/.cache/huggingface/token
- Has saved token ?: True
- Who am I ?: Wauplin
- Configured git credential helpers: store
- FastAI: N/A
- Torch: 1.12.1
- Jinja2: 3.1.2
- Graphviz: 0.20.1
- Pydot: 1.4.2
- Pillow: 9.2.0
- hf_transfer: 0.1.3
- gradio: 4.0.2
- tensorboard: 2.6
- numpy: 1.23.2
- pydantic: 2.4.2
- aiohttp: 3.8.4
- ENDPOINT: https://huggingface.co
- HF_HUB_CACHE: /home/wauplin/.cache/huggingface/hub
- HF_ASSETS_CACHE: /home/wauplin/.cache/huggingface/assets
- HF_TOKEN_PATH: /home/wauplin/.cache/huggingface/token
- HF_HUB_OFFLINE: False
- HF_HUB_DISABLE_TELEMETRY: False
- HF_HUB_DISABLE_PROGRESS_BARS: None
- HF_HUB_DISABLE_SYMLINKS_WARNING: False
- HF_HUB_DISABLE_EXPERIMENTAL_WARNING: False
- HF_HUB_DISABLE_IMPLICIT_TOKEN: False
- HF_HUB_ENABLE_HF_TRANSFER: False
- HF_HUB_ETAG_TIMEOUT: 10
- HF_HUB_DOWNLOAD_TIMEOUT: 10
```
