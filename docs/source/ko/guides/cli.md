<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 명령줄 인터페이스 (CLI) [[command-line-interface]]

`huggingface_hub` Python 패키지는 `huggingface-cli`라는 내장 CLI를 함께 제공합니다. 이 도구를 사용하면 터미널에서 Hugging Face Hub와 직접 상호 작용할 수 있습니다. 계정에 로그인하고, 리포지토리를 생성하고, 파일을 업로드 및 다운로드하는 등의 다양한 작업을 수행할 수 있습니다. 또한 머신을 구성하거나 캐시를 관리하는 데 유용한 기능도 제공합니다. 이 가이드는 CLI의 주요 기능과 사용 방법에 관해 설명합니다.

## 시작하기 [[getting-started]]

먼저, CLI를 설치해 보세요:

```
>>> pip install -U "huggingface_hub[cli]"
```

<Tip>

위의 코드에서 사용자 경험을 높이기 위해 `[cli]` 추가 종속성을 포함하였습니다. 이는 `delete-cache` 명령을 사용할 때 특히 유용합니다.

</Tip>

설치가 완료되면, CLI가 올바르게 설정되었는지 확인할 수 있습니다:

```
>>> huggingface-cli --help
usage: huggingface-cli <command> [<args>]

positional arguments:
  {env,login,whoami,logout,repo,upload,download,lfs-enable-largefiles,lfs-multipart-upload,scan-cache,delete-cache}
                        huggingface-cli command helpers
    env                 Print information about the environment.
    login               Log in using a token from huggingface.co/settings/tokens
    whoami              Find out which huggingface.co account you are logged in as.
    logout              Log out
    repo                {create} Commands to interact with your huggingface.co repos.
    upload              Upload a file or a folder to a repo on the Hub
    download            Download files from the Hub
    lfs-enable-largefiles
                        Configure your repository to enable upload of files > 5GB.
    scan-cache          Scan cache directory.
    delete-cache        Delete revisions from the cache directory.

options:
  -h, --help            show this help message and exit
```

CLI가 제대로 설치되었다면 CLI에서 사용 가능한 모든 옵션 목록이 출력됩니다. `command not found: huggingface-cli`와 같은 오류 메시지가 표시된다면 [설치](../installation) 가이드를 확인하세요.

<Tip>

`--help` 옵션을 사용하면 명령어에 대한 자세한 정보를 얻을 수 있습니다. 언제든지 사용 가능한 모든 옵션과 그 세부 사항을 확인할 수 있습니다. 예를 들어 `huggingface-cli upload --help`는 CLI를 사용하여 파일을 업로드하는 구체적인 방법을 알려줍니다.

</Tip>

### 다른 방법으로 설치하기 [[alternative-install]]

#### pkgx 사용하기 [[using-pkgx]]

[Pkgx](https://pkgx.sh)는 다양한 플랫폼에서 빠르게 작동하는 패키지 매니저입니다. 다음과 같이 pkgx를 사용하여 huggingface-cli를 설치할 수 있습니다:

```bash
>>> pkgx install huggingface-cli
```

또는 pkgx를 통해 huggingface-cli를 직접 실행할 수도 있습니다:

```bash
>>> pkgx huggingface-cli --help
```

pkgx huggingface에 대한 자세한 내용은 [여기](https://pkgx.dev/pkgs/huggingface.co/)에서 확인할 수 있습니다.

#### Homebrew 사용하기 [[using-homebrew]]

[Homebrew](https://brew.sh/)를 사용하여 CLI를 설치할 수도 있습니다:

```bash
>>> brew install huggingface-cli
```

Homebrew huggingface에 대한 자세한 내용은 [여기](https://formulae.brew.sh/formula/huggingface-cli)에서 확인할 수 있습니다.

## huggingface-cli login [[huggingface-cli-login]]

Hugging Face Hub에 접근하는 대부분의 작업(비공개 리포지토리 액세스, 파일 업로드, PR 제출 등)을 위해서는 Hugging Face 계정에 로그인해야 합니다. 로그인을 하기 위해서 [설정 페이지](https://huggingface.co/settings/tokens)에서 생성한 [사용자 액세스 토큰](https://huggingface.co/docs/hub/security-tokens)이 필요하며, 이 토큰은 Hub에서의 사용자 인증에 사용됩니다. 파일 업로드나 콘텐츠 수정을 위해선 쓰기 권한이 있는 토큰이 필요합니다.
토큰을 받은 후에 터미널에서 다음 명령을 실행하세요:

```bash
>>> huggingface-cli login
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
>>> huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
```

[이 단락](../quick-start#authentication)에서 인증에 대한 더 자세한 내용을 확인할 수 있습니다.

## huggingface-cli whoami [[huggingface-cli-whoami]]

로그인 여부를 확인하기 위해 `huggingface-cli whoami` 명령어를 사용할 수 있습니다. 이 명령어는 옵션이 없으며, 간단하게 사용자 이름과 소속된 조직들을 출력합니다:

```bash
huggingface-cli whoami
Wauplin
orgs:  huggingface,eu-test,OAuthTesters,hf-accelerate,HFSmolCluster
```

로그인하지 않은 경우 오류 메시지가 출력됩니다.

## huggingface-cli logout [[huggingface-cli-logout]]

이 명령어를 사용하여 로그아웃할 수 있습니다. 실제로는 컴퓨터에 저장된 토큰을 삭제합니다.

하지만 `HF_TOKEN` 환경 변수를 사용하여 로그인했다면, 이 명령어로는 로그아웃할 수 없습니다([참조]((../package_reference/environment_variables#hftoken))). 대신 컴퓨터의 환경 설정에서 `HF_TOKEN` 변수를 제거하면 됩니다.

## huggingface-cli download [[huggingface-cli-download]]


`huggingface-cli download` 명령어를 사용하여 Hub에서 직접 파일을 다운로드할 수 있습니다. [다운로드](./download) 가이드에서 설명된 [`hf_hub_download`], [`snapshot_download`] 헬퍼 함수를 사용하여 반환된 경로를 터미널에 출력합니다. 우리는 아래 예시에서 가장 일반적인 사용 사례를 살펴볼 것입니다. 사용 가능한 모든 옵션을 보려면 아래 명령어를 실행해보세요:

```bash
huggingface-cli download --help
```

### 파일 한 개 다운로드하기 [[download-a-single-file]]

리포지토리에서 파일 하나를 다운로드하고 싶다면, repo_id와 다운받고 싶은 파일명을 아래와 같이 입력하세요:

```bash
>>> huggingface-cli download gpt2 config.json
downloading https://huggingface.co/gpt2/resolve/main/config.json to /home/wauplin/.cache/huggingface/hub/tmpwrq8dm5o
(…)ingface.co/gpt2/resolve/main/config.json: 100%|██████████████████████████████████| 665/665 [00:00<00:00, 2.49MB/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

이 명령어를 실행하면 항상 마지막 줄에 파일 경로를 출력합니다.

### 전체 리포지토리 다운로드하기 [[download-an-entire-repository]]

리포지토리의 모든 파일을 다운로드하고 싶을 때에는 repo id만 입력하면 됩니다:

```bash
>>> huggingface-cli download HuggingFaceH4/zephyr-7b-beta
Fetching 23 files:   0%|                                                | 0/23 [00:00<?, ?it/s]
...
...
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### 여러 파일 다운로드하기 [[download-multiple-files]]

리포지토리의 전체 폴더를 다운로드하지 않고 한 번에 여러 파일을 다운로드할 수도 있습니다. 이를 위한 두 가지 방법이 있습니다. 다운로드하고자 하는 파일들의 목록이 정해져 있다면, 해당 파일명을 순서대로 입력하면 됩니다:

```bash
>>> huggingface-cli download gpt2 config.json model.safetensors
Fetching 2 files:   0%|                                                                        | 0/2 [00:00<?, ?it/s]
downloading https://huggingface.co/gpt2/resolve/11c5a3d5811f50298f278a704980280950aedb10/model.safetensors to /home/wauplin/.cache/huggingface/hub/tmpdachpl3o
(…)8f278a7049802950aedb10/model.safetensors: 100%|██████████████████████████████| 8.09k/8.09k [00:00<00:00, 40.5MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.76it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

또 다른 방법은 `--include`와 `--exclude` 옵션을 사용하여 원하는 파일을 필터링하는 것입니다. 예를 들어, [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)의 모든 safetensors 파일을 다운로드하되 FP16 정밀도의 파일은 제외하고 싶다면 다음과 같이 실행할 수 있습니다:

```bash
>>> huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*
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
>>> huggingface-cli download HuggingFaceH4/ultrachat_200k --repo-type dataset

# https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat
>>> huggingface-cli download HuggingFaceH4/zephyr-chat --repo-type space

...
```

### 특정 리비전 다운로드하기 [[download-a-specific-revision]]

따로 리비전을 지정하지 않는다면 기본적으로 main 브랜치의 최신 커밋에서 파일을 다운로드합니다. 특정 리비전(커밋 해시, 브랜치 이름 또는 태그)에서 다운로드하려면 `--revision` 옵션을 사용하세요:

```bash
>>> huggingface-cli download bigcode/the-stack --repo-type dataset --revision v1.1
...
```

### 로컬 폴더에 다운로드하기 [[download-to-a-local-folder]]

Hub에서 파일을 다운로드하는 권장되고 기본적인 방법은 캐시 시스템을 사용하는 것입니다. 그러나 특정한 경우에는 파일을 지정된 폴더로 다운로드하고 옮기고 싶을 수 있습니다. 이는 git 명령어와 유사한 워크플로우를 만드는데 도움이 됩니다. `--local_dir` 옵션을 사용하여 이 작업을 수행할 수 있습니다.

<Tip warning={true}>

로컬 폴더에 다운로드하는 것에는 몇 가지 단점이 있습니다. `--local-dir` 명령어를 사용하기 전에 [다운로드](./download#download-files-to-local-folder) 가이드에서 해당 내용을 확인해보세요.

</Tip>

```bash
>>> huggingface-cli download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir .
...
./model-00001-of-00002.safetensors
```

### 캐시 디렉터리 지정하기 [[specify-cache-directory]]

기본적으로 모든 파일은 `HF_HOME` [환경 변수](../package_reference/environment_variables#hfhome)에서 정의한 캐시 디렉터리에 다운로드됩니다. `--cache-dir`을 사용하여 직접 캐시 위치를 지정할 수 있습니다:

```bash
>>> huggingface-cli download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### 토큰 설정하기 [[specify-a-token]]

비공개 또는 접근이 제한된 리포지토리들에 접근하기 위해서는 토큰이 필요합니다. 기본적으로 로컬에 저장된 토큰(`huggingface-cli login`)이 사용됩니다. 직접 인증하고 싶다면 `--token` 옵션을 사용해보세요:

```bash
>>> huggingface-cli download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### 조용한 모드 [[quiet-mode]]

`huggingface-cli download` 명령은 상세한 정보를 출력합니다. 경고 메시지, 다운로드된 파일 정보, 진행률 등이 포함됩니다. 이 모든 출력을 숨기려면 `--quiet` 옵션을 사용하세요. 이 옵션을 사용하면 다운로드된 파일의 경로가 표시되는 마지막 줄만 출력됩니다. 이 기능은 스크립트에서 다른 명령어로 출력을 전달하고자 할 때 유용할 수 있습니다.

```bash
>>> huggingface-cli download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

## huggingface-cli upload [[huggingface-cli-upload]]

`huggingface-cli upload` 명령어로 Hub에 직접 파일을 업로드할 수 있습니다. [업로드](./upload) 가이드에서 설명된 [`upload_file`], [`upload_folder`] 헬퍼 함수를 사용합니다. 우리는 아래 예시에서 가장 일반적인 사용 사례를 살펴볼 것입니다. 사용 가능한 모든 옵션을 보려면 아래 명령어를 실행해보세요:

```bash
>>> huggingface-cli upload --help
```

### 전체 폴더 업로드하기 [[upload-an-entire-folder]]

이 명령어의 기본 사용법은 다음과 같습니다:

```bash
# Usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
```

현재 디텍터리를 리포지토리의 루트 위치에 업로드하려면, 아래 명령어를 사용하세요:

```bash
>>> huggingface-cli upload my-cool-model . .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

<Tip>

리포지토리가 아직 존재하지 않으면 자동으로 생성됩니다.

</Tip>

또한, 특정 폴더만 업로드하는 것도 가능합니다:

```bash
>>> huggingface-cli upload my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

마지막으로, 리포지토리의 특정 위치에 폴더를 업로드할 수 있습니다:

```bash
>>> huggingface-cli upload my-cool-model ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/my-cool-model/tree/main/data/train
```

### 파일 한 개 업로드하기 [[upload-a-single-file]]

컴퓨터에 있는 파일을 가리키도록 `local_path`를 설정함으로써 파일 한 개를 업로드할 수 있습니다. 이때, `path_in_repo`는 선택사항이며 로컬 파일 이름을 기본값으로 사용합니다:

```bash
>>> huggingface-cli upload Wauplin/my-cool-model ./models/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors
```

파일 한 개를 특정 디렉터리에 업로드하고 싶다면, `path_in_repo`를 그에 맞게 설정하세요:

```bash
>>> huggingface-cli upload Wauplin/my-cool-model ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/vae/model.safetensors
```

### 여러 파일 업로드하기 [[upload-multiple-files]]

전체 폴더를 업로드하지 않고 한 번에 여러 파일을 업로드하려면 `--include`와 `--exclude` 옵션을 사용해보세요. 리포지토리에 있는 파일을 삭제하면서 새 파일을 업로드하는 `--delete` 옵션과 함께 사용할 수 있습니다. 아래 예시는 `/logs` 안의 파일을 제외한 모든 파일을 업로드하고 원격 파일들을 삭제함으로써 로컬 Space를 동기화하는 방법을 보여줍니다:

```bash
# Sync local Space with Hub (upload new files except from logs/, delete removed files)
>>> huggingface-cli upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
...
```

### 데이터 세트 또는 Space에 업로드하기 [[upload-to-a-dataset-or-space]]

데이터 세트나 Space에 업로드하려면 `--repo-type` 옵션을 사용하세요:

```bash
>>> huggingface-cli upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
...
```

### 조직에 업로드하기 [[upload-to-an-organization]]

개인 리포지토리 대신 조직이 소유한 리포지토리에 파일을 업로드하려면 `repo_id`를 입력해야 합니다:

```bash
>>> huggingface-cli upload MyCoolOrganization/my-cool-model . .
https://huggingface.co/MyCoolOrganization/my-cool-model/tree/main/
```

### 특정 개정에 업로드하기 [[upload-to-a-specific-revision]]

기본적으로 파일은 `main` 브랜치에 업로드됩니다. 다른 브랜치나 참조에 파일을 업로드하려면 `--revision` 옵션을 사용하세요:

```bash
# Upload files to a PR
>>> huggingface-cli upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**참고:** `revision`이 존재하지 않고 `--create-pr` 옵션이 설정되지 않은 경우, `main` 브랜치에서 자동으로 새 브랜치가 생성됩니다.

### 업로드 및 PR 생성하기 [[upload-and-create-a-pr]]

리포지토리에 푸시할 권한이 없다면, PR을 생성하여 작성자들에게 변경하고자 하는 내용을 알려야 합니다. 이를 위해서 `--create-pr` 옵션을 사용할 수 있습니다:

```bash
# Create a PR and upload the files to it
>>> huggingface-cli upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### 정기적으로 업로드하기 [[upload-at-regular-intervals]]

리포지토리에 정기적으로 업데이트하고 싶을 때, `--every` 옵션을 사용할 수 있습니다. 예를 들어, 모델을 훈련하는 중에 로그 폴더를 10분마다 업로드하고 싶다면 다음과 같이 사용할 수 있습니다:

```bash
# Upload new logs every 10 minutes
huggingface-cli upload training-model logs/ --every=10
```

### 커밋 메시지 지정하기 [[specify-a-commit-message]]

`--commit-message`와 `--commit-description`을 사용하여 기본 메시지 대신 사용자 지정 메시지와 설명을 커밋에 설정하세요:

```bash
>>> huggingface-cli upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### 토큰 지정하기 [[specify-a-token]]

파일을 업로드하려면 토큰이 필요합니다. 기본적으로 로컬에 저장된 토큰(`huggingface-cli login`)이 사용됩니다. 직접 인증하고 싶다면 `--token` 옵션을 사용해보세요:

```bash
>>> huggingface-cli upload Wauplin/my-cool-model ./models . --token=hf_****
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### 조용한 모드 [[quiet-mode]]

기본적으로 `huggingface-cli upload` 명령은 상세한 정보를 출력합니다. 경고 메시지, 업로드된 파일 정보, 진행률 등이 포함됩니다. 이 모든 출력을 숨기려면 `--quiet` 옵션을 사용하세요. 이 옵션을 사용하면 업로드된 파일의 URL이 표시되는 마지막 줄만 출력됩니다. 이 기능은 스크립트에서 다른 명령어로 출력을 전달하고자 할 때 유용할 수 있습니다.

```bash
>>> huggingface-cli upload Wauplin/my-cool-model ./models . --quiet
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

## huggingface-cli scan-cache [[huggingface-cli-scan-cache]]

캐시 디렉토리를 스캔하여 다운로드한 리포지토리가 무엇인지와 디스크에서 차지하는 공간을 알 수 있습니다. `huggingface-cli scan-cache` 명령어를 사용하여 이를 확인해보세요:

```bash
>>> huggingface-cli scan-cache
REPO ID                     REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS                LOCAL PATH
--------------------------- --------- ------------ -------- ------------- ------------- ------------------- -------------------------------------------------------------------------
glue                        dataset         116.3K       15 4 days ago    4 days ago    2.4.0, main, 1.17.0 /home/wauplin/.cache/huggingface/hub/datasets--glue
google/fleurs               dataset          64.9M        6 1 week ago    1 week ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/datasets--google--fleurs
Jean-Baptiste/camembert-ner model           441.0M        7 2 weeks ago   16 hours ago  main                /home/wauplin/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
bert-base-cased             model             1.9G       13 1 week ago    2 years ago                       /home/wauplin/.cache/huggingface/hub/models--bert-base-cased
t5-base                     model            10.1K        3 3 months ago  3 months ago  main                /home/wauplin/.cache/huggingface/hub/models--t5-base
t5-small                    model           970.7M       11 3 days ago    3 days ago    refs/pr/1, main     /home/wauplin/.cache/huggingface/hub/models--t5-small

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

캐시 디렉토리 스캔에 대한 자세한 내용을 알고 싶다면, [캐시 관리](./manage-cache#scan-cache-from-the-terminal) 가이드를 확인해보세요.

## huggingface-cli delete-cache [[huggingface-cli-delete-cache]]

사용하지 않는 캐시를 삭제하고 싶다면 `huggingface-cli delete-cache`를 사용해보세요. 이는 디스크 공간을 절약하고 확보하는 데 유용합니다. 이에 대한 자세한 내용은 [캐시 관리](./manage-cache#clean-cache-from-the-terminal) 가이드에서 확인할 수 있습니다.

## huggingface-cli env [[huggingface-cli-env]]

`huggingface-cli env` 명령어는 사용자의 컴퓨터 설정에 대한 상세한 정보를 보여줍니다. 이는 [GitHub](https://github.com/huggingface/huggingface_hub)에서 문제를 제출할 때, 관리자가 문제를 파악하고 해결하는 데 도움이 됩니다.

```bash
>>> huggingface-cli env

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
- Tensorflow: 2.11.0
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
