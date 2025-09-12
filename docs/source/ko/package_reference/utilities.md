<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 유틸리티[[utilities]]

## 로깅 구성[[huggingface_hub.utils.logging.get_verbosity]]

`huggingface_hub` 패키지는 패키지 로그 레벨을 제어하기 위한 `logging` 유틸리티를 제공합니다. 
다음과 같이 가져올 수 있습니다:

```py
from huggingface_hub import logging
```

그런 다음, 로그의 출력 수를 업데이트하기 위해 로그 레벨을 정의할 수 있습니다:

```python
from huggingface_hub import logging

logging.set_verbosity_error()
logging.set_verbosity_warning()
logging.set_verbosity_info()
logging.set_verbosity_debug()

logging.set_verbosity(...)
```

로그 레벨은 다음과 같이 이해하면 됩니다:

- `error`: 오류 또는 예기치 않은 동작으로 이어질 수 있는 결정적인 로그만 표시합니다.
- `warning`:  결정적이진 않지만 의도치 않은 동작을 초래할 수 있는 로그를 표시합니다. 또한 중요한 정보를 포함한 로그도 표시될 수 있습니다.
- `info`: 하부에서 무슨 일이 일어나고 있는지에 대한 자세한 로그를 포함하여 대부분의 로그를 표시합니다. 무언가 예상치 못한 방식으로 동작하는 경우, 더 많은 정보를 얻기 위해 verbosity 단계로 전환하는 것이 좋습니다.
- `debug`: 하부에서 정확히 무슨 일이 일어나고 있는지를 추적하는 데 사용될 수 있는 일부 내부 로그를 포함하여 모든 로그를 표시합니다.

[[autodoc]] logging.get_verbosity
[[autodoc]] logging.set_verbosity
[[autodoc]] logging.set_verbosity_info
[[autodoc]] logging.set_verbosity_debug
[[autodoc]] logging.set_verbosity_warning
[[autodoc]] logging.set_verbosity_error
[[autodoc]] logging.disable_propagation
[[autodoc]] logging.enable_propagation

### 리포지토리별 도우미 메소드[[huggingface_hub.utils.logging.get_logger]]

아래 제공된 메소드들은 `huggingface_hub` 라이브러리 모듈을 수정할 때 관련이 있습니다. `huggingface_hub`를 사용하고 해당 모듈을 수정하지 않는 경우에는 사용할 필요가 없습니다.

[[autodoc]] logging.get_logger

## 프로그레스 바 구성하기[[configure-progress-bars]]

프로그레스 바는 긴 시간이 걸리는 작업을 실행하는 동안 정보를 표시하는 유용한 도구입니다(예시로 파일을 다운로드하거나 업로드하는 등). `huggingface_hub`는 라이브러리 전체에서 일관된 방식으로 프로그레스 바를 표시하기 위한 [`~utils.tqdm`] 래퍼를 제공합니다.

기본적으로 프로그레스 바가 활성화되어 있습니다. `HF_HUB_DISABLE_PROGRESS_BARS` 환경 변수를 설정하여 전역적으로 비활성화할 수 있습니다. 또한 [`~utils.enable_progress_bars`]와 [`~utils.disable_progress_bars`]를 사용하여 프로그레스 바를 개별적으로 활성화 또는 비활성화할 수도 있습니다. 만약 환경 변수가 설정되어 있다면, 환경 변수가 도우미에서 우선 순위를 가집니다.


```py
>>> from huggingface_hub import snapshot_download
>>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

>>> # 전역적으로 프로그레스 바를 비활성화합니다.
>>> disable_progress_bars()

>>> # 프로그레스 바가 표시되지 않습니다!
>>> snapshot_download("gpt2")

>>> are_progress_bars_disabled()
True

>>> # 다시 프로그레스 바가 활성화됩니다
>>> enable_progress_bars()
```

### are_progress_bars_disabled[[huggingface_hub.utils.are_progress_bars_disabled]]

[[autodoc]] huggingface_hub.utils.are_progress_bars_disabled

### disable_progress_bars[[huggingface_hub.utils.disable_progress_bars]]

[[autodoc]] huggingface_hub.utils.disable_progress_bars

### enable_progress_bars[huggingface_hub.utils.enable_progress_bars]]

[[autodoc]] huggingface_hub.utils.enable_progress_bars


## HTTP 오류 다루기[[handle-http-errors]]

`huggingface_hub`는 서버에서 반환된 추가 정보로 `requests`에서 발생한 `HTTPError`를 세분화하기 위해 자체 HTTP 오류를 정의합니다.

### 예외 발생[[huggingface_hub.utils.hf_raise_for_status]]

[`~utils.hf_raise_for_status`]는 Hub에 대한 모든 요청에서 "상태를 확인하고 예외를 발생시키는" 중앙 메소드로 사용됩니다. 이 메서드는 기본 `requests.raise_for_status`를 감싸서 추가 정보를 제공합니다. 발생된 모든 `HTTPError`는 `HfHubHTTPError`로 변환됩니다.

```py
import requests
from huggingface_hub.utils import hf_raise_for_status, HfHubHTTPError

response = requests.post(...)
try:
    hf_raise_for_status(response)
except HfHubHTTPError as e:
    print(str(e)) # 형식화된 메시지
    e.request_id, e.server_message # 서버에서 반환된 세부 정보

    # 오류 메시지를 발생시킬 때 추가 정보를 포함하여 완성합니다
    e.append_to_message("\n`create_commit` expects the repository to exist.")
    raise
```

[[autodoc]] huggingface_hub.utils.hf_raise_for_status

### HTTP 오류[[http-errors]]

여기에는 `huggingface_hub`에서 발생하는 HTTP 오류 목록이 있습니다.

#### HfHubHTTPError[[huggingface_hub.errors.HfHubHTTPError]]

`HfHubHTTPError`는 HF Hub HTTP 오류에 대한 부모 클래스입니다. 이 클래스는 서버 응답을 구문 분석하고 오류 메시지를 형식화하여 사용자에게 가능한 많은 정보를 제공합니다.

[[autodoc]] huggingface_hub.errors.HfHubHTTPError

#### RepositoryNotFoundError[[huggingface_hub.errors.RepositoryNotFoundError]]

[[autodoc]] huggingface_hub.errors.RepositoryNotFoundError

#### GatedRepoError[[huggingface_hub.errors.GatedRepoError]]

[[autodoc]] huggingface_hub.errors.GatedRepoError

#### RevisionNotFoundError[[huggingface_hub.errors.RevisionNotFoundError]]

[[autodoc]] huggingface_hub.errors.RevisionNotFoundError

#### BadRequestError[[huggingface_hub.errors.BadRequestError]]

[[autodoc]] huggingface_hub.errors.BadRequestError

#### EntryNotFoundError[[huggingface_hub.errors.EntryNotFoundError]]

[[autodoc]] huggingface_hub.errors.EntryNotFoundError

#### RemoteEntryNotFoundError[[huggingface_hub.errors.RemoteEntryNotFoundError]]

[[autodoc]] huggingface_hub.errors.RemoteEntryNotFoundError

#### LocalEntryNotFoundError[[huggingface_hub.errors.LocalEntryNotFoundError]]

[[autodoc]] huggingface_hub.errors.LocalEntryNotFoundError

#### OfflineModeIsEnabledd[[huggingface_hub.errors.OfflineModeIsEnabled]]

[[autodoc]] huggingface_hub.errors.OfflineModeIsEnabled

## 원격 측정[[huggingface_hub.utils.send_telemetry]]

`huggingface_hub`는 원격 측정 데이터를 보내는 도우미가 포함되어 있습니다. 이 정보는 문제를 디버깅하고 새로운 기능을 우선적으로 처리하는 데 도움이 됩니다. 사용자는 `HF_HUB_DISABLE_TELEMETRY=1` 환경 변수를 설정하여 언제든지 원격 측정 수집을 비활성화할 수 있습니다. 또한 오프라인 모드에서도 (즉, HF_HUB_OFFLINE=1로 설정된 경우) 원격 측정이 비활성화됩니다.

서드 파티 라이브러리의 유지 관리자인 경우, 원격 측정 데이터를 보내는 것은 [`send_telemetry`]를 호출하는 것만큼 간단합니다. 사용자에게 가능한 영향을 최소화하기 위해 데이터는 별도의 스레드에서 전송됩니다.

[[autodoc]] utils.send_telemetry


## 검증기[[validators]]

`huggingface_hub`에는 메소드 인수를 자동으로 유효성 검사하는 사용자 정의 검증기가 포함되어 있습니다. 이 유효성 검사는 타입 힌트를 검증하는 데 [Pydantic](https://pydantic-docs.helpmanual.io/)의 작업을 참고하여 구현되었지만, 기능은 더 제한적입니다.

### 일반 데코레이터[[generic-decorator]]

[`~utils.validate_hf_hub_args`]는 `huggingface_hub`의 네이밍을 따르는 인수를 갖는 메소드를 캡슐화하는 일반적인 데코레이터입니다. 기본적으로 구현된 검증기가 있는 모든 인수가 유효성 검사됩니다.

입력이 유효하지 않은 경우 [`~utils.HFValidationError`]이 발생합니다. 첫 번째 유효하지 않은 값만 오류를 발생시키고 유효성 검사 프로세스를 중지합니다.

사용법:

```py
>>> from huggingface_hub.utils import validate_hf_hub_args

>>> @validate_hf_hub_args
... def my_cool_method(repo_id: str):
...     print(repo_id)

>>> my_cool_method(repo_id="valid_repo_id")
valid_repo_id

>>> my_cool_method("other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> my_cool_method(repo_id="other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
```

#### validate_hf_hub_args[[huggingface_hub.utils.validate_hf_hub_args]]

[[autodoc]] utils.validate_hf_hub_args

#### HFValidationError[[huggingface_hub.utils.HFValidationError]]

[[autodoc]] utils.HFValidationError

### Argument validators[[argument-validators]]

검증기는 개별적으로도 사용할 수 있습니다. 다음은 검증할 수 있는 모든 인수 목록입니다.

#### repo_id[[huggingface_hub.utils.validate_repo_id]]

[[autodoc]] utils.validate_repo_id
