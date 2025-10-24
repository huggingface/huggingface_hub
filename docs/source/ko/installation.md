<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 설치 방법 [[installation]]

시작하기 전에 적절한 패키지를 설치하여 환경을 설정해야 합니다.

`huggingface_hub`는 **Python 3.9+**에서 테스트되었습니다.

## pip로 설치하기 [[install-with-pip]]

[가상 환경](https://docs.python.org/3/library/venv.html)에서 `huggingface_hub`를 설치하는 것을 적극 권장합니다.
파이썬 가상 환경에 익숙하지 않다면 이 [가이드](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)를 참고하세요.
가상 환경을 사용하면 여러 프로젝트를 더 쉽게 관리하고 의존성 간의 호환성 문제를 피할 수 있습니다.

프로젝트 디렉토리에 가상 환경을 생성하는 것으로 시작하세요:

```bash
python -m venv .env
```

가상환경을 활성화하려면 Linux 및 macOS의 경우:

```bash
source .env/bin/activate
```

Windows의 경우:

```bash
.env/Scripts/activate
```

[PyPi 레지스트리](https://pypi.org/project/huggingface-hub/)에서 `huggingface_hub`를 설치할 준비가 되었습니다:

```bash
pip install --upgrade huggingface_hub
```

완료되면 [설치 확인](#check-installation)이 올바르게 작동하는지 확인합니다.

### 선택 의존성 설치 [[install-optional-dependencies]]

`huggingface_hub`의 일부 의존성은 `huggingface_hub`의 핵심 기능을 실행하는 데 필요하지 않으므로 [선택적](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies)입니다. 설치가 되어있지 않다면 `huggingface_hub`의 추가적인 기능을 사용하지 못할 수 있습니다.

선택적 의존성은 `pip`을 통해 설치할 수 있습니다:
```bash
# PyTorch와 CLI와 관련된 기능에 대한 의존성을 모두 설치합니다.
pip install 'huggingface_hub[cli,torch]'
```

다음은 `huggingface_hub`의 선택 의존성 목록입니다:
- `cli`: 보다 편리한 `huggingface_hub`의 CLI 인터페이스입니다.
- `fastai`, `torch`: 프레임워크별 기능을 실행하려면 필요합니다.
- `dev`: 라이브러리에 기여하고 싶다면 필요합니다. 테스트 실행을 위한 `testing`, 타입 검사기 실행을 위한 `typing`, 린터 실행을 위한 `quality`가 포함됩니다.

### 소스에서 설치 [[install-from-source]]

경우에 따라 소스에서 직접 `huggingface_hub`를 설치하는 게 더 나을수도 있습니다.
이렇게 하면 최신 릴리스 버전이 아닌 최신 `main` 버전을 사용할 수 있습니다.
`main` 버전은 마지막 공식 릴리스 이후 버그가 수정되었지만 아직 새 릴리스가 출시되지 않은 경우와 같이 최신 개발 사항을 들고오는 데 유용합니다.

동시에 `main` 버전은 항상 안정적일 수 없다는 뜻이기도 합니다. 저희는 `main` 버전을 계속 운영하기 위해 노력하고 있으며, 대부분의 문제는 보통 몇 시간 또는 하루 이내에 해결됩니다. 문제가 발생하면 이슈를 열어주시면 더 빨리 해결할 수 있어요!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

소스에서 설치할 때 특정 브랜치를 지정할 수도 있습니다. 아직 병합되지 않은 새로운 기능이나 새로운 버그 수정을 테스트하려는 경우에 유용합니다:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

완료되면 [설치 확인](#check-installation)을 통해 올바르게 작동하는지 확인하세요.

### 편집 가능한 설치 [[editable-install]]
소스에서 설치하면 [편집 가능한 설치](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)를 설정할 수 있습니다.
이런 고급 설치는 `huggingface_hub`에 기여하고 코드의 변경 사항을 테스트해야 하는 경우에 쓰입니다. 컴퓨터에 `huggingface_hub`의 로컬 복사본을 클론해둬야 합니다.

```bash
# 먼저 로컬에 리포지토리를 복제하세요.
git clone https://github.com/huggingface/huggingface_hub.git

# 그런 다음 -e 플래그를 사용하여 설치하세요.
cd huggingface_hub
pip install -e .
```

이렇게 클론한 레포지토리 폴더와 Python 경로를 연결합니다.
이제 Python은 일반적인 라이브러리 경로 외에도 복제된 폴더 내부를 찾습니다.
예를 들어 파이썬 패키지가 일반적으로 `./.venv/lib/python3.13/site-packages/`에 설치되어 있다면, Python은 복제된 폴더 `./huggingface_hub/`도 검색하게 됩니다.

## conda로 설치하기 [[install-with-conda]]

이미 익숙하다면 [conda-forge 채널](https://anaconda.org/conda-forge/huggingface_hub)를 통해 `huggingface_hub`를 설치할 수도 있습니다:


```bash
conda install -c conda-forge huggingface_hub
```

완료되면 [설치 확인](#check-installation)을 통해 올바르게 작동하는지 확인하세요.

## 설치 확인 [[check-installation]]

설치가 완료되면 다음 명령을 실행하여 `huggingface_hub`가 제대로 작동하는지 확인하세요:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

이 명령은 Hub에서 [gpt2](https://huggingface.co/gpt2) 모델에 대한 정보를 가져옵니다.
출력은 다음과 같아야 합니다:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows 제한 사항 [[windows-limitations]]

좋은 ML을 어디서나 사용할 수 있게 하자는 목표 아래, `huggingface_hub`를 크로스 플랫폼 라이브러리로 만들었으며, 특히 유닉스 기반과 Windows 시스템 모두에서 잘 작동하도록 했습니다. 그럼에도 `huggingface_hub`를 Windows에서 실행할 때 몇 가지 제한이 있습니다. 다음은 알려진 문제의 전체 목록입니다. 문서화되지 않은 문제가 발생하면 [GitHub에 이슈](https://github.com/huggingface/huggingface_hub/issues/new/choose)를 열어서 알려주시기 바랍니다.

- `huggingface_hub`의 캐시 시스템은 Hub에서 다운로드한 파일을 효율적으로 캐시하기 위해 심볼릭 링크에 의존합니다. Windows에서는 개발자 모드를 활성화하거나 관리자 권한으로 스크립트를 실행해야 심볼릭 링크를 활성화할 수 있습니다. 활성화하지 않으면 캐시 시스템이 계속 작동하지만 최적화되지 않은 방식으로 작동합니다. 자세한 내용은 [캐시 제한](./guides/manage-cache#limitations) 섹션을 참조하세요.
- Hub의 파일 경로에는 특수 문자를 사용할 수 있습니다(예: `"path/to?/my/file"`). 드문 경우이길 바라지만, Windows는 [특수 문자](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)에 대한 제한이 더 엄격하기 때문에 해당 파일을 다운로드할 수 없습니다. 실수라고 생각되면 레포지토리 소유자에게 문의하시거나 해결책을 찾기 위해 저희에게 연락해 주세요.


## 다음 단계 [[next-steps]]

컴퓨터에 `huggingface_hub`가 제대로 설치되면 [환경 변수를 설정](package_reference/environment_variables)하거나 [가이드 중 하나를 골라](guides/overview) 시작할 수 있습니다.
