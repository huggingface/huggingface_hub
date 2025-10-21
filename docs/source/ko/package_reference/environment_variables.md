<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 환경 변수[[environment-variables]]

`huggingface_hub`는 환경 변수를 사용해 설정할 수 있습니다.

환경 변수에 대해 잘 알지 못하다면 그에 대한 문서인 [macOS and Linux](https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/)와 
[Windows](https://phoenixnap.com/kb/windows-set-environment-variable)를 참고하세요.

이 문서에서는 `huggingface_hub`와 관련된 모든 환경 변수와 그 의미에 대해 안내합니다.

## 일반적인 변수[[generic]]

### HF_INFERENCE_ENDPOINT[[hfinferenceendpoint]]

추론 API 기본 URL을 구성합니다. 조직에서 추론 API를 직접 가리키는 것이 아니라 API 게이트웨이를 가리키는 경우 이 변수를 설정할 수 있습니다.

기본값은 `"https://api-inference.huggingface.co"`입니다.

### HF_HOME[[hfhome]]

`huggingface_hub`가 어디에 데이터를 로컬로 저장할 지 위치를 구성합니다. 특히 토큰과 캐시가 이 폴더에 저장됩니다.

[XDG_CACHE_HOME](#xdgcachehome)이 설정되어 있지 않다면, 기본값은 `"~/.cache/huggingface"`입니다.

### HF_HUB_CACHE[[hfhubcache]]

Hub의 리포지토리가 로컬로 캐시될 위치(모델, 데이터세트 및 스페이스)를 구성합니다.

기본값은 `"$HF_HOME/hub"` (예로 들면, 기본 설정은 `"~/.cache/huggingface/hub"`)입니다.

### HF_ASSETS_CACHE[[hfassetscache]]

다운스트림 라이브러리에서 생성된 [assets](../guides/manage-cache#caching-assets)가 로컬로 캐시되는 위치를 구성합니다.
이 assets은 전처리된 데이터, GitHub에서 다운로드한 파일, 로그, ... 등이 될 수 있습니다.

기본값은 `"$HF_HOME/assets"` (예로 들면, 기본 설정은 `"~/.cache/huggingface/assets"`)입니다.

### HF_TOKEN[[hftoken]]

Hub에 인증하기 위한 사용자 액세스 토큰을 구성합니다. 이 값을 설정하면 머신에 저장된 토큰을 덮어씁니다(`$HF_TOKEN_PATH`, 또는 `$HF_TOKEN_PATH`가 설정되지 않은 경우 `"$HF_HOME/token"`에 저장됨).

인증에 대한 자세한 내용은 [이 섹션](../quick-start#인증)을 참조하세요.

### HF_TOKEN_PATH[[hftokenpath]]

`huggingface_hub`가 사용자 액세스 토큰(User Access Token)을 저장할 위치를 구성합니다. 기본값은 `"$HF_HOME/token"`(예로 들면, 기본 설정은 `~/.cache/huggingface/token`)입니다.

### HF_HUB_VERBOSITY[[hfhubverbosity]]

`huggingface_hub`의 로거(logger)의 상세도 수준(verbosity level)을 설정합니다. 다음 중 하나여야 합니다.
`{"debug", "info", "warning", "error", "critical"}` 중 하나여야 합니다.

기본값은 `"warning"`입니다.

더 자세한 정보를 알아보고 싶다면, [logging reference](../package_reference/utilities#huggingface_hub.utils.logging.get_verbosity)를 살펴보세요.

### HF_HUB_ETAG_TIMEOUT[[hfhubetagtimeout]]

파일을 다운로드하기 전에 리포지토리에서 최신 메타데이터를 가져올 때 서버 응답을 기다리는 시간(초)을 정의하는 정수 값입니다. 요청 시간이 초과되면 `huggingface_hub`는 기본적으로 로컬에 캐시된 파일을 사용합니다. 값을 낮게 설정하면 이미 파일을 캐시한 연결 속도가 느린 컴퓨터의 워크플로 속도가 빨라집니다. 값이 클수록 더 많은 경우에서 메타데이터 호출이 성공할 수 있습니다. 기본값은 10초입니다.

### HF_HUB_DOWNLOAD_TIMEOUT[[hfhubdownloadtimeout]]

파일을 다운로드할 때 서버 응답을 기다리는 시간(초)을 정의하는 정수 값입니다. 요청 시간이 초과되면 TimeoutError가 발생합니다. 연결 속도가 느린 컴퓨터에서는 값을 높게 설정하는 것이 좋습니다. 값이 작을수록 네트워크가 완전히 중단된 경우에 프로세스가 더 빨리 실패합니다. 기본값은 10초입니다.

## 불리언 값[[boolean-values]]

다음 환경 변수는 불리언 값을 요구합니다. 변수는 값이 `{"1", "ON", "YES", "TRUE"}`(대소문자 구분 없음) 중 하나이면 `True`로 간주합니다. 다른 값(또는 정의되지 않음)은 `False`로 간주됩니다.

### HF_HUB_OFFLINE[[hfhuboffline]]

이 옵션을 설정하면 Hugging Face Hub에 HTTP 호출이 이루어지지 않습니다. 파일을 다운로드하려고 하면 캐시된 파일만 액세스됩니다. 캐시 파일이 감지되지 않으면 오류를 발생합니다. 네트워크 속도가 느리고 파일의 최신 버전이 중요하지 않은 경우에 유용합니다.

환경 변수로 `HF_HUB_OFFLINE=1`이 설정되어 있고 [`HfApi`]의 메소드를 호출하면 [`~huggingface_hub.utils.OfflineModeIsEnabled`] 예외가 발생합니다.

**참고:** 최신 버전의 파일이 캐시되어 있더라도 `hf_hub_download`를 호출하면 새 버전을 사용할 수 없는지 확인하기 위해 HTTP 요청이 발생합니다. `HF_HUB_OFFLINE=1`을 설정하면 이 호출을 건너뛰어 로딩 시간이 빨라집니다.

### HF_HUB_DISABLE_IMPLICIT_TOKEN[[hfhubdisableimplicittoken]]

Hub에 대한 모든 요청이 반드시 인증을 필요로 하는 것은 아닙니다. 예를 들어 `"gpt2"` 모델에 대한 세부 정보를 요청하는 경우에는 인증이 필요하지 않습니다. 그러나 사용자가 [로그인](../package_reference/login) 상태인 경우, 기본 동작은 사용자 경험을 편하게 하기 위해 비공개 또는 게이트 리포지토리에 액세스할 때 항상 토큰을 전송하는 것(HTTP 401 권한 없음이 표시되지 않음)입니다. 개인 정보 보호를 위해 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`로 설정하여 이 동작을 비활성화할 수 있습니다. 이 경우 토큰은 "쓰기 권한" 호출(예: 커밋 생성)에만 전송됩니다.

**참고:** 토큰을 항상 전송하는 것을 비활성화하면 이상한 부작용이 발생할 수 있습니다. 예를 들어 Hub에 모든 모델을 나열하려는 경우 당신의 비공개 모델은 나열되지 않습니다. 사용자 스크립트에 명시적으로 `token=True` 인수를 전달해야 합니다.

### HF_HUB_DISABLE_PROGRESS_BARS[[hfhubdisableprogressbars]]

시간이 오래 걸리는 작업의 경우 `huggingface_hub`는 기본적으로 진행률 표시줄을 표시합니다(tqdm 사용).
모든 진행률 표시줄을 한 번에 비활성화하려면 `HF_HUB_DISABLE_PROGRESS_BARS=1`으로 설정하면 됩니다.

### HF_HUB_DISABLE_SYMLINKS_WARNING[[hfhubdisablesymlinkswarning]]

Windows 머신을 사용하는 경우 개발자 모드를 활성화하거나 관리자 모드에서 `huggingface_hub`를 관리자 모드로 실행하는 것이 좋습니다. 그렇지 않은 경우 `huggingface_hub`가 캐시 시스템에 심볼릭 링크를 생성할 수 없습니다. 모든 스크립트를 실행할 수 있지만 일부 대용량 파일이 하드 드라이브에 중복될 수 있으므로 사용자 경힘이 저하될 수 있습니다. 이 동작을 경고하기 위해 경고 메시지가 나타납니다. `HF_HUB_DISABLE_SYMLINKS_WARNING=1`로 설정하면 이 경고를 비활성화할 수 있습니다.

자세한 내용은 [캐시 제한](../guides/manage-cache#limitations)을 참조하세요.

### HF_HUB_DISABLE_EXPERIMENTAL_WARNING[[hfhubdisableexperimentalwarning]]

`huggingface_hub`의 일부 기능은 실험 단계입니다. 즉, 사용은 가능하지만 향후 유지될 것이라고 보장할 수는 없습니다. 특히 이러한 기능의 API나 동작은 지원 중단 없이 업데이트될 수 있습니다. 실험적 기능을 사용할 때는 이에 대한 경고를 위해 경고 메시지가 나타납니다. 실험적 기능을 사용하여 잠재적인 문제를 디버깅하는 것이 편하다면 `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1`으로 설정하여 경고를 비활성화할 수 있습니다.

실험적인 기능을 사용 중이라면 알려주세요! 여러분의 피드백은 기능을 설계하고 개선하는 데 도움이 됩니다.

### HF_HUB_DISABLE_TELEMETRY[[hfhubdisabletelemetry]]

기본적으로 일부 데이터는 사용량을 모니터링하고 문제를 디버그하며 기능의 우선순위를 정하는 데 도움을 주기 위해 HF 라이브러리(`transformers`, `datasets`, `gradio`,...)에서 수집합니다. 각 라이브러리는 자체 정책(즉, 모니터링할 사용량)을 정의하지만 핵심 구현은 `huggingface_hub`에서 이루어집니다([`send_telemetry`] 참조).

환경 변수로 `HF_HUB_DISABLE_TELEMETRY=1`을 설정하여 원격 측정을 전역적으로 비활성화할 수 있습니다.

### HF_HUB_ENABLE_HF_TRANSFER[[hfhubenablehftransfer]]

Hub에서 `hf_transfer`를 사용하여 더 빠르게 업로드 및 다운로드하려면 `True`로 설정하세요.
기본적으로 `huggingface_hub`는 파이썬 기반 `requests.get` 및 `requests.post` 함수를 사용합니다.
이 함수들은 안정적이고 다용도로 사용할 수 있지만 대역폭이 높은 머신에서는 가장 효율적인 선택이 아닐 수 있습니다. [`hf_transfer`](https://github.com/huggingface/hf_transfer)는 대용량 파일을 작은 부분으로 분할하여 사용 대역폭을 최대화하고
여러 스레드를 사용하여 동시에 전송함으로써 대역폭을 최대화하기 위해 개발된 Rust 기반 패키지입니다. 이 접근 방식은 전송 속도를 거의 두 배로 높일 수 있습니다.
`hf_transfer`를 사용하려면:

1. `huggingface_hub`를 설치할 때 `hf_transfer`를 추가로 지정합니다.
   (예시: `pip install huggingface_hub[hf_transfer]`).
2. 환경 변수로 `HF_HUB_ENABLE_HF_TRANSFER=1`을 설정합니다.

`hf_transfer`를 사용하면 특정 제한 사항이 있다는 점에 유의하세요. 순수 파이썬 기반이 아니므로 오류 디버깅이 어려울 수 있습니다. 또한 `hf_transfer`에는 다운로드 재개 및 프록시와 같은 몇 가지 사용자 친화적인 기능이 없습니다. 이런 부족한 부분은 Rust 로직의 단순성과 속도를 유지하기 위해 의도한 것입니다. 이런 이유들로, `hf_transfer`는 `huggingface_hub`에서 기본적으로 활성화되지 않습니다.

## 사용되지 않는 환경 변수[[deprecated-environment-variables]]

Hugging Face 생태계의 모든 환경 변수를 표준화하기 위해 일부 변수는 사용되지 않는 것으로 표시되었습니다. 해당 변수는 여전히 작동하지만 더 이상 대체한 변수보다 우선하지 않습니다. 다음 표에는 사용되지 않는 변수와 해당 대체 변수가 간략하게 설명되어 있습니다:

| 사용되지 않는 변수          | 대체 변수          |
| --------------------------- | ------------------ |
| `HUGGINGFACE_HUB_CACHE`     | `HF_HUB_CACHE`     |
| `HUGGINGFACE_ASSETS_CACHE`  | `HF_ASSETS_CACHE`  |
| `HUGGING_FACE_HUB_TOKEN`    | `HF_TOKEN`         |
| `HUGGINGFACE_HUB_VERBOSITY` | `HF_HUB_VERBOSITY` |

## 외부 도구[[from-external-tools]]

일부 환경 변수는 `huggingface_hub`에만 특정되지는 않지만 설정 시 함께 고려됩니다.

### DO_NOT_TRACK[[donottrack]]

불리언 값입니다. `hf_hub_disable_telemetry`에 해당합니다. True로 설정하면 Hugging Face Python 생태계(`transformers`, `diffusers`, `gradio` 등)에서 원격 측정이 전역적으로 비활성화됩니다. 자세한 내용은 https://consoledonottrack.com/ 을 참조하세요.

### NO_COLOR[[nocolor]]

불리언 값입니다. 이 값을 설정하면 `hf` 도구는 ANSI 색상을 출력하지 않습니다. [no-color.org](https://no-color.org/)를 참조하세요.

### XDG_CACHE_HOME[[xdgcachehome]]

`HF_HOME`이 설정되지 않은 경우에만 사용합니다!

이것은 Linux 시스템에서 [사용자별 비필수(캐시된) 데이터](https://wiki.archlinux.org/title/XDG_Base_Directory)가 쓰여져야 하는 위치를 구성하는 기본 방법입니다.

`HF_HOME`이 설정되지 않은 경우 기본 홈은 `"~/.cache/huggingface"`대신  `"$XDG_CACHE_HOME/huggingface"`가 됩니다.
