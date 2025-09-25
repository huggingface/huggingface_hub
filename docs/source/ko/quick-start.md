<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 둘러보기 [[quickstart]]

[Hugging Face Hub](https://huggingface.co/)는 머신러닝 모델, 데모, 데이터 세트 및 메트릭을 공유할 수 있는 곳입니다. `huggingface_hub` 라이브러리는 개발 환경을 벗어나지 않고도 Hub와 상호작용할 수 있도록 도와줍니다. 리포지토리를 쉽게 만들고 관리하거나, 파일을 다운로드 및 업로드하고, 유용한 모델과 데이터 세트의 메타데이터도 구할 수 있습니다.

## 설치 [[installation]]

시작하려면 `huggingface_hub` 라이브러리를 설치하세요:

```bash
pip install --upgrade huggingface_hub
```

자세한 내용은 [설치](./installation) 가이드를 참조하세요.

## 파일 다운로드 [[download-files]]

Hub의 리포지토리는 git으로 버전 관리되며, 사용자는 단일 파일 또는 전체 리포지토리를 다운로드할 수 있습니다. 파일을 다운로드하려면 [`hf_hub_download`] 함수를 사용하면 됩니다.
사용하면 파일을 다운로드하여 로컬 디스크에 캐시하기 때문에, 다음에 해당 파일이 필요하면 캐시에서 가져오므로 다시 다운로드할 필요가 없습니다.

다운로드하려면 리포지토리 ID와 파일명이 필요합니다. 예를 들어, [Pegasus](https://huggingface.co/google/pegasus-xsum) 모델 구성 파일을 다운로드하려면:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

특정 버전의 파일을 다운로드하려면 `revision` 매개변수를 사용하여 브랜치 이름, 태그 또는 커밋 해시를 지정하세요. 커밋 해시를 사용하기로 선택한 경우, 7자로 된 짧은 커밋 해시 대신 전체 길이의 해시여야 합니다:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

자세한 내용과 옵션은 [`hf_hub_download`]에 대한 API 레퍼런스를 참조하세요.

## 로그인 [[login]]

비공개 리포지토리 다운로드, 파일 업로드, PR 생성 등 Hub와 상호 작용하려면 Hugging Face 계정으로 로그인해야 하는 경우가 많습니다.
아직 계정이 없다면 [계정 만들기](https://huggingface.co/join)를 클릭한 다음, 로그인하여 [설정 페이지](https://huggingface.co/settings/tokens)에서 [사용자 액세스 토큰](https://huggingface.co/docs/hub/security-tokens)을 받으세요. 사용자 액세스 토큰은 Hub에 인증하는 데 사용됩니다.

사용자 액세스 토큰을 받으면 터미널에서 다음 명령을 실행하세요:

```bash
hf auth login
# or using an environment variable
hf auth login --token $HUGGINGFACE_TOKEN
```

또는 주피터 노트북이나 스크립트에서 [`login`]로 프로그래밍 방식으로 로그인할 수도 있습니다:

```py
>>> from huggingface_hub import login
>>> login()
```

`login(token="hf_xxx")`과 같이 토큰을 [`login`]에 직접 전달하여 토큰을 입력하라는 메시지를 표시하지 않고 프로그래밍 방식으로 로그인할 수도 있습니다. 이렇게 한다면 소스 코드를 공유할 때 주의하세요. 토큰을 소스코드에 명시적으로 저장하는 대신에 보안 저장소에서 토큰을 가져오는 것이 가장 좋습니다.

한 번에 하나의 계정에만 로그인할 수 있습니다. 새 계정으로 로그인하면 이전 계정에서 로그아웃됩니다. 항상 `hf auth whoami` 명령으로 어떤 계정을 사용 중인지 확인하세요.
동일한 스크립트에서 여러 계정을 처리하려면 각 메서드를 호출할 때 토큰을 제공하면 됩니다. 이 방법은 머신에 토큰을 저장하지 않으려는 경우에도 유용합니다.

> [!WARNING]
> 로그인하면 Hub에 대한 모든 요청(반드시 인증이 필요하지 않은 메소드 포함)은 기본적으로 액세스 토큰을 사용합니다. 토큰의 암시적 사용을 비활성화하려면 `HF_HUB_DISABLE_IMPLICIT_TOKEN` 환경 변수를 설정해야 합니다.

## 리포지토리 만들기 [[create-a-repository]]

등록 및 로그인이 완료되면 [`create_repo`] 함수를 사용하여 리포지토리를 생성하세요:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

리포지토리를 비공개로 설정하려면 다음과 같이 하세요:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

비공개 리포지토리는 본인 외에는 누구에게도 공개되지 않습니다.

> [!TIP]
> 리포지토리를 생성하거나 Hub에 콘텐츠를 푸시하려면 `write` (쓰기) 권한이 있는 사용자 액세스 토큰을 제공해야 합니다. 토큰을 생성할 때 [설정 페이지](https://huggingface.co/settings/tokens)에서 권한을 선택할 수 있습니다.

## 파일 업로드 [[upload-files]]

새로 만든 리포지토리에 파일을 추가하려면 [`upload_file`] 함수를 사용하세요. 다음을 지정해야 합니다:

1. 업로드할 파일의 경로
2. 리포지토리에 있는 파일의 경로
3. 파일을 추가할 위치의 리포지토리 ID

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

한 번에 두 개 이상의 파일을 업로드하려면 [업로드](./guides/upload) 가이드에서 (git을 포함하거나 제외한) 여러 가지 파일 업로드 방법을 소개하는 가이드를 참조하세요.

## 다음 단계 [[next-steps]]

`huggingface_hub` 라이브러리는 사용자가 파이썬으로 Hub와 상호작용할 수 있는 쉬운 방법을 제공합니다. Hub에서 파일과 리포지토리를 관리하는 방법에 대해 자세히 알아보려면 [How-to 가이드](./guides/overview)를 읽어보시기 바랍니다:

- 보다 쉽게 [리포지토리를 관리](./guides/repository)해보세요.
- Hub에서 [다운로드](./guides/download) 파일을 다운로드해보세요.
- Hub에 [업로드](./guides/upload) 파일을 업로드해보세요.
- 원하는 모델 또는 데이터 세트에 대한 [Hub에서 검색](./guides/search)해보세요.
- 빠른 추론을 원하신다면 [추론 API](./guides/inference)를 사용해보세요.
