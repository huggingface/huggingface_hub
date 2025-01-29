<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 리포지토리 생성과 관리[[create-and-manage-a-repository]]

Hugging Face Hub는 Git 리포지토리 모음입니다. [Git](https://git-scm.com/)은 협업을 할 때 여러 프로젝트 버전을 쉽게 관리하기 위해 널리 사용되는 소프트웨어 개발 도구입니다. 이 가이드에서는 Hub의 리포지토리 사용법인 다음 내용을 다룹니다:

- 리포지토리 생성과 삭제.
- 태그 및 브랜치 관리.
- 리포지토리 이름 변경.
- 리포지토리 공개 여부.
- 리포지토리 복사본 관리.

<Tip warning={true}>

GitLab/GitHub/Bitbucket과 같은 플랫폼을 사용해 본 경험이 있다면, 모델 리포지토리를 관리하기 위해 `git` CLI를 사용해 git 리포지토리를 클론(`git clone`)하고 변경 사항을 커밋(`git add, git commit`)하고 커밋한 내용을 푸시(`git push`) 하는것이 가장 먼저 떠오를 것입니다. 이 명령어들은 Hugging Face Hub에서도 사용할 수 있습니다. 하지만 소프트웨어 엔지니어링과 머신러닝은 동일한 요구 사항과 워크플로우를 공유하지 않습니다. 모델 리포지토리는 다양한 프레임워크와 도구를 위한 대규모 모델 가중치 파일을 유지관리 할 수 있으므로, 리포지토리를 복제하면 대규모 로컬 폴더를 유지관리하고 막대한 크기의 파일을 다루게 될 수 있습니다. 결과적으로 Hugging Face의 커스텀 HTTP 방법을 사용하는 것이 더욱 효율적일 수 있습니다. 더 자세한 내용은 [Git vs HTTP paradigm](../concepts/git_vs_http) 문서를 참조하세요.

</Tip>

Hub에 리포지토리를 생성하고 관리하려면, 로그인이 되어 있어야 합니다. 로그인이 안 되어있다면 [이 문서](../quick-start#authentication)를 참고해 주세요. 이 가이드에서는 로그인이 되어있다는 가정하에 진행됩니다.

## 리포지토리 생성 및 삭제[[repo-creation-and-deletion]]

첫 번째 단계는 어떻게 리포지토리를 생성하고 삭제하는지를 알아야 합니다. 사용자 이름 네임스페이스 아래에 소유한 리포지토리 또는 쓰기 권한이 있는 조직의 리포지토리만 관리할 수 있습니다.

### 리포지토리 생성[[create-a-repository]]

[`create_repo`] 함수로 함께 빈 리포지토리를 만들고 `repo_id` 매개변수를 사용하여 이름을 정하세요. `repo_id`는 사용자 이름 또는 조직 이름 뒤에 리포지토리 이름이 따라옵니다: `username_or_org/repo_name`.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

기본적으로 [`create_repo`]는 모델 리포지토리를 만듭니다. 하지만 `repo_type` 매개변수를 사용하여 다른 유형의 리포지토리를 지정할 수 있습니다. 예를 들어 데이터셋 리포지토리를 만들고 싶다면:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

리포지토리를 만들 때, `private` 매개변수를 사용하여 가시성을 설정할 수 있습니다.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", private=True)
```

추후 리포지토리 가시성을 변경하고 싶다면, [`update_repo_settings`] 함수를 이용해 바꿀 수 있습니다.

### 리포지토리 삭제[[delete-a-repository]]

[`delete_repo`]를 사용하여 리포지토리를 삭제할 수 있습니다. 리포지토리를 삭제하기 전에 신중히 결정하세요. 왜냐하면, 삭제하고 나서 다시 되돌릴 수 없는 프로세스이기 때문입니다!

삭제하려는 리포지토리의 `repo_id`를 지정하세요:

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

### 리포지토리 복제(Spaces 전용)[[duplicate-a-repository-only-for-spaces]]

가끔 다른 누군가의 리포지토리를 복사하여, 상황에 맞게 수정하고 싶을 때가 있습니다. 이는 [`duplicate_space`]를 사용하여 Space에 복사할 수 있습니다. 이 함수를 사용하면 리포지토리 전체를 복제할 수 있습니다. 그러나 여전히 하드웨어, 절전 시간, 리포지토리, 변수 및 비밀번호와 같은 자체 설정을 구성해야 합니다. 자세한 내용은 [Manage your Space](./manage-spaces) 문서를 참조하십시오.

```py
>>> from huggingface_hub import duplicate_space
>>> duplicate_space("multimodalart/dreambooth-training", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

## 파일 다운로드와 업로드[[upload-and-download-files]]

이제 리포지토리를 생성했으므로, 변경 사항을 푸시하고 파일을 다운로드하는 것에 관심이 있을 것입니다.

이 두 가지 주제는 각각 자체 가이드가 필요합니다. 리포지토리 사용하는 방법에 대해 알아보려면 [업로드](./upload) 및 [다운로드](./download) 문서를 참조하세요.

## 브랜치와 태그[[branches-and-tags]]

Git 리포지토리는 동일한 리포지토리의 다른 버전을 저장하기 위해 브랜치들을 사용합니다. 태그는 버전을 출시할 때와 같이 리포지토리의 특정 상태를 표시하는 데 사용될 수도 있습니다. 일반적으로 브랜치와 태그는 [git 참조](https://git-scm.com/book/en/v2/Git-Internals-Git-References)
로 참조됩니다.

### 브랜치 생성과 태그[[create-branches-and-tags]]

[`create_branch`]와 [`create_tag`]를 이용하여 새로운 브랜치와 태그를 생성할 수 있습니다.

```py
>>> from huggingface_hub import create_branch, create_tag

# `main` 브랜치를 기반으로 Space 저장소에 새 브랜치를 생성합니다.
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# `v0.1-release` 브랜치를 기반으로 Dataset 저장소에 태그를 생성합니다.
>>> create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

같은 방식으로 [`delete_branch`]와 [`delete_tag`] 함수를 사용하여 브랜치 또는 태그를 삭제할 수 있습니다.

### 모든 브랜치와 태그 나열[[list-all-branches-and-tags]]

[`list_repo_refs`]를 사용하여 리포지토리로부터 현재 존재하는 git 참조를 나열할 수 있습니다:

```py
>>> from huggingface_hub import list_repo_refs
>>> list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## 리포지토리 설정 변경[[change-repository-settings]]

리포지토리는 구성할 수 있는 몇 가지 설정이 있습니다. 대부분의 경우 브라우저의 리포지토리 설정 페이지에서 직접 설정할 것입니다. 설정을 바꾸려면 리포지토리에 대한 쓰기 액세스 권한이 있어야 합니다(사용자 리포지토리거나, 조직의 구성원이어야 함). 이 주제에서는 `huggingface_hub`를 사용하여 프로그래밍 방식으로 구성할 수 있는 설정을 알아보겠습니다.

Spaces를 위한 특정 설정들(하드웨어, 환경변수 등)을 구성하기 위해서는 [Manage your Spaces](../guides/manage-spaces) 문서를 참조하세요.

### 가시성 업데이트[[update-visibility]]

리포지토리는 공개 또는 비공개로 설정할 수 있습니다. 비공개 리포지토리는 해당 저장소의 사용자 혹은 소속된 조직의 구성원만 볼 수 있습니다. 다음과 같이 리포지토리를 비공개로 변경할 수 있습니다.

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

### 리포지토리 이름 변경[[rename-your-repository]]

[`move_repo`]를 사용하여 Hub에 있는 리포지토리 이름을 변경할 수 있습니다. 이 함수를 사용하여 개인에서 조직 리포지토리로 이동할 수도 있습니다. 이렇게 하면 [일부 제한 사항](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo)이 있으므로 주의해야 합니다. 예를 들어, 다른 사용자에게 리포지토리를 이전할 수는 없습니다.

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

## 리포지토리의 로컬 복사본 관리[[manage-a-local-copy-of-your-repository]]

위에 설명한 모든 작업은 HTTP 요청을 사용하여 작업할 수 있습니다. 그러나 경우에 따라 로컬 복사본을 가지고 익숙한 Git 명령어를 사용하여 상호 작용하는 것이 편리할 수 있습니다.

[`Repository`] 클래스는 Git 명령어와 유사한 기능을 제공하는 함수를 사용하여 Hub의 파일 및 리포지토리와 상호 작용할 수 있습니다. 이는 이미 알고 있고 좋아하는 Git 및 Git-LFS 방법을 사용하는 래퍼(wrapper)입니다. 시작하기 전에 Git-LFS가 설치되어 있는지 확인하세요([여기서](https://git-lfs.github.com/) 설치 지침을 확인할 수 있습니다).

<Tip warning={true}>

[`Repository`]는 [`HfApi`]에 구현된 HTTP 기반 대안을 선호하여 중단되었습니다. 아직 많은 레거시 코드에서 사용되고 있기 때문에 [`Repository`]가 완전히 제거되는 건 `v1.0` 릴리스에서만 이루어집니다. 자세한 내용은 [해당 설명 페이지](./concepts/git_vs_http)를 참조하세요.

</Tip>

### 로컬 리포지토리 사용[[use-a-local-repository]]

로컬 리포지토리 경로를 사용하여 [`Repository`] 객체를 생성하세요:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="<path>/<to>/<folder>")
```

### 복제[[clone]]

`clone_from` 매개변수는 Hugging Face 리포지토리 ID에서 로컬 디렉터리로 리포지토리를 복제합니다. 이때 `local_dir` 매개변수를 사용하여 로컬 디렉터리에 저장합니다:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="w2v2", clone_from="facebook/wav2vec2-large-960h-lv60")
```

`clone_from`은 URL을 사용해 리포지토리를 복제할 수 있습니다.

```py
>>> repo = Repository(local_dir="huggingface-hub", clone_from="https://huggingface.co/facebook/wav2vec2-large-960h-lv60")
```

`clone_from` 매개변수를 [`create_repo`]와 결합하여 리포지토리를 만들고 복제할 수 있습니다.

```py
>>> repo_url = create_repo(repo_id="repo_name")
>>> repo = Repository(local_dir="repo_local_path", clone_from=repo_url)
```

리포지토리를 복제할 때 `git_user` 및 `git_email` 매개변수를 지정함으로써 복제한 리포지토리에 Git 사용자 이름과 이메일을 설정할 수 있습니다. 사용자가 해당 리포지토리에 커밋하면 Git은 커밋 작성자를 인식합니다.

```py
>>> repo = Repository(
...   "my-dataset",
...   clone_from="<user>/<dataset_id>",
...   token=True,
...   repo_type="dataset",
...   git_user="MyName",
...   git_email="me@cool.mail"
... )
```

### 브랜치[[branch]]

브랜치는 현재 코드와 파일에 영향을 미치지 않으면서 협업과 실험에 중요합니다.[`~Repository.git_checkout`]을 사용하여 브랜치 간에 전환할 수 있습니다. 예를 들어, `branch1`에서 `branch2`로 전환하려면:

```py
>>> from huggingface_hub import Repository
>>> repo = Repository(local_dir="huggingface-hub", clone_from="<user>/<dataset_id>", revision='branch1')
>>> repo.git_checkout("branch2")
```

### 끌어오기[[pull]]

[`~Repository.git_pull`]은 원격 리포지토리로부터의 변경사항을 현재 로컬 브랜치에 업데이트하게 합니다.

```py
>>> from huggingface_hub import Repository
>>> repo.git_pull()
```

브랜치가 원격에서의 새 커밋으로 업데이트 된 후에 로컬 커밋을 수행하고자 한다면 `rebase=True`를 설정하세요:

```py
>>> repo.git_pull(rebase=True)
```
