<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# `huggingface_hub` 캐시 시스템 관리하기[[manage-huggingfacehub-cache-system]]

## 캐싱 이해하기[[understand-caching]]

Hugging Face Hub 캐시 시스템은 Hub에 의존하는 라이브러리 간에 공유되는 중앙 캐시로 설계되었습니다. v0.8.0에서 수정한 파일 간에 다시 다운로드하는 것을 방지하도록 업데이트되었습니다.

캐시 시스템은 다음과 같이 설계되었습니다:

```
<CACHE_DIR>
├─ <MODELS>
├─ <DATASETS>
├─ <SPACES>
```

`<CACHE_DIR>`는 보통 사용자의 홈 디렉토리입니다. 그러나 모든 메소드에서 `cache_dir` 인수를 사용하거나 `HF_HOME` 또는 `HF_HUB_CACHE` 환경 변수를 지정하여 사용자 정의할 수 있습니다.

모델, 데이터셋, 스페이스는 공통된 루트를 공유합니다. 각 리포지토리는 리포지토리 유형과 네임스페이스(조직 또는 사용자 이름이 있을 경우), 리포지토리 이름을 포함합니다:

```
<CACHE_DIR>
├─ models--julien-c--EsperBERTo-small
├─ models--lysandrejik--arxiv-nlp
├─ models--bert-base-cased
├─ datasets--glue
├─ datasets--huggingface--DataMeasurementsFiles
├─ spaces--dalle-mini--dalle-mini
```

Hub로부터 모든 파일이 이 폴더들 안에 다운로드됩니다. 캐싱은 파일이 이미 존재하고 업데이트되지 않은 경우, 파일을 두 번 다운로드하지 않도록 해줍니다.
하지만 파일이 업데이트되었고 최신 파일을 요청하면, 최신 파일을 다운로드합니다 (이전 파일은 그대로 유지되어 필요할 때 다시 사용할 수 있습니다).

이를 위해 모든 폴더는 동일한 구조를 가집니다:

```
<CACHE_DIR>
├─ datasets--glue
│  ├─ refs
│  ├─ blobs
│  ├─ snapshots
...
```

각 폴더는 다음과 같은 내용을 포함하도록 구성되었습니다:

### Refs[[refs]]

`refs` 폴더에는 주어진 참조의 최신 수정 버전을 나타내는 파일이 포함되어 있습니다. 예를 들어, 이전에 리포지토리의 `main` 브랜치에서 파일을 가져온 경우, `refs` 폴더에는 `main`이라는 이름의 파일이 포함되며, 이 파일 자체에는 현재 헤드의 커밋 식별자가 들어 있습니다.

만약 `main`의 최신 커밋 식별자가 `aaaaaa`라면, 그 파일에는 `aaaaaa`가 들어 있습니다.

같은 브랜치가 새로운 커밋으로 업데이트되어 `bbbbbb`라는 식별자를 갖게 되면, 해당 참조에서 파일을 다시 다운로드할 때 `refs/main` 파일은 `bbbbbb`로 업데이트됩니다.

### Blobs[[blobs]]

`blobs` 폴더에는 실제로 다운로드된 파일이 포함되어 있습니다. 각 파일의 이름은 해당 파일의 해시값입니다.

### Snapshots[[snapshots]]

`snapshots` 폴더에는 위에서 언급한 blobs에 대한 심볼릭 링크가 포함되어 있습니다. 이 폴더는 여러 개의 하위 폴더로 구성되어 있으며, 각 폴더는 알려진 수정 버전을 나타냅니다.

위 설명에서, 처음에 `aaaaaa` 버전에서 파일을 가져왔고, 그 후에 `bbbbbb` 버전에서 파일을 가져왔습니다. 이 상황에서 `snapshots` 폴더에는 `aaaaaa`와 `bbbbbb`라는 두 개의 폴더가 있습니다.

이 폴더들 각각에는 다운로드한 파일의 이름을 가진 심볼릭 링크가 있습니다. 예를 들어, `aaaaaa` 버전에서 `README.md` 파일을 다운로드했다면, 다음과 같은 경로가 생깁니다:

```
<CACHE_DIR>/<REPO_NAME>/snapshots/aaaaaa/README.md
```

그 `README.md` 파일은 실제로 해당 파일의 해시를 가진 blob에 대한 심볼릭 링크입니다.

이와 같은 구조를 생성함으로써 파일 공유 메커니즘이 열리게 됩니다. 동일한 파일을 `bbbbbb` 버전에서 가져온 경우, 동일한 해시를 가지게 되어 파일을 다시 다운로드할 필요가 없습니다.

### .no_exist (advanced)[[noexist-advanced]]

`blobs`, `refs`, `snapshots` 폴더 외에도 캐시에서 `.no_exist` 폴더를 찾을 수 있습니다. 이 폴더는 한 번 다운로드하려고 시도했지만 Hub에 존재하지 않는 파일을 기록합니다. 이 폴더의 구조는 `snapshots` 폴더와 동일하며, 알려진 각 수정 버전에 대해 하나의 하위 폴더를 갖습니다:

```
<CACHE_DIR>/<REPO_NAME>/.no_exist/aaaaaa/config_that_does_not_exist.json
```
`snapshots` 폴더와 달리, 파일은 단순히 빈 파일입니다 (심볼릭 링크가 아님). 이 예에서 `"config_that_does_not_exist.json"` 파일은 `"aaaaaa"` 버전에 대해 Hub에 존재하지 않습니다. 빈 파일만 저장하므로, 이 폴더는 디스크 사용량을 크게 차지하지 않기에 무시할 수 있습니다.

그렇다면 이제 여러분은 왜 이 정보가 관련이 있는지 궁금해 할지도 모릅니다. 몇몇 경우에서는 프레임워크가 모델에 대한 옵션 파일들을 불러오려고 시도합니다. 존재하지 않는 옵션 파일들을 저장하면 가능한 옵션 파일당 1개의 HTTP 호출을 절약할 수 있어 모델을 더 빠르게 불러올 수 있습니다. 이는 예를 들어 각 토크나이저가 추가 파일을 지원하는 `transformers`에서 발생합니다. 처음으로 토크나이저를 로드할 때, 다음 초기화를 위해 로딩 시간을 더 빠르게 하기 위해 옵션 파일이 존재하는지 여부를 캐시합니다.

HTTP 요청을 만들지 않고 로컬로 캐시된 파일이 있는지 테스트하려면, [`try_to_load_from_cache`] 헬퍼를 사용할 수 있습니다. 이것은 파일이 존재하고 캐시된 경우에는 파일 경로를, 존재하지 않음이 캐시된 경우에는 `_CACHED_NO_EXIST` 객체를, 알 수 없는 경우에는 `None`을 반환합니다.

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

filepath = try_to_load_from_cache()
if isinstance(filepath, str):
    # 파일이 존재하고 캐시됩니다
    ...
elif filepath is _CACHED_NO_EXIST:
    # 파일의 존재여부가 캐시됩니다
    ...
else:
    # 파일은 캐시되지 않습니다
    ...
```

### 캐시 구조 예시[[in-practice]]

실제로는 캐시는 다음과 같은 트리 구조를 가질 것입니다:

```text
    [  96]  .
    └── [ 160]  models--julien-c--EsperBERTo-small
        ├── [ 160]  blobs
        │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
        ├── [  96]  refs
        │   └── [  40]  main
        └── [ 128]  snapshots
            ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
            │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
            │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
            └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
                ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
                └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
```

### 제한사항[[limitations]]

효율적인 캐시 시스템을 갖기 위해 `huggingface-hub`은 심볼릭 링크를 사용합니다. 그러나 모든 기기에서 심볼릭 링크를 지원하지는 않습니다. 특히 Windows에서 이러한 한계가 있다는 것이 알려져 있습니다. 이런 경우에는 `huggingface_hub`이 `blobs/` 디렉터리를 사용하지 않고 대신 파일을 직접 `snapshots/` 디렉터리에 저장합니다. 이 해결책을 통해 사용자는 Hub에서 파일을 다운로드하고 캐시하는 방식을 정확히 동일하게 사용할 수 있습니다. 캐시를 검사하고 삭제하는 도구들도 지원됩니다. 그러나 캐시 시스템은 동일한 리포지토리의 여러 수정 버전을 다운로드하는 경우 같은 파일이 여러 번 다운로드될 수 있기 때문에 효율적이지 않을 수 있습니다.

Windows 기기에서 심볼릭 링크 기반 캐시 시스템의 이점을 누리려면, [개발자 모드를 활성화](https://docs.microsoft.com/ko-kr/windows/apps/get-started/enable-your-device-for-development)하거나 Python을 관리자 권한으로 실행해야 합니다.

심볼릭 링크가 지원되지 않는 경우, 사용자에게 캐시 시스템의 낮은 버전을 사용 중임을 알리는 경고 메시지가 표시됩니다. 이 경고는 `HF_HUB_DISABLE_SYMLINKS_WARNING` 환경 변수를 true로 설정하여 비활성화할 수 있습니다.

## 캐싱 자산[[caching-assets]]

Hub에서 파일을 캐시하는 것 외에도, 하위 라이브러리들은 종종 `huggingface_hub`에 직접 처리되지 않는 HF와 관련된 다른 파일을 캐시해야 할 때가 있습니다 (예: GitHub에서 다운로드한 파일, 전처리된 데이터, 로그 등). 이러한 파일, 즉 '자산(assets)'을 캐시하기 위해 [`cached_assets_path`]를 사용할 수 있습니다. 이 헬퍼는 요청한 라이브러리의 이름과 선택적으로 네임스페이스 및 하위 폴더 이름을 기반으로 HF 캐시의 경로를 통일된 방식으로 생성합니다. 목표는 모든 하위 라이브러리가 자산을 자체 방식대로(예: 구조에 대한 규칙 없음) 관리할 수 있도록 하는 것입니다. 그러나 올바른 자산 폴더 내에 있어야 합니다. 그러한 라이브러리는 `huggingface_hub`의 도구를 활용하여 캐시를 관리할 수 있으며, 특히 CLI 명령을 통해 자산의 일부를 스캔하고 삭제할 수 있습니다.

```py
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # 자산 폴더에서 원하는 대로 작업하세요!
```

> [!TIP]
> [`cached_assets_path`]는 자산을 저장하는 권장 방법이지만 필수는 아닙니다. 이미 라이브러리가 자체 캐시를 사용하는 경우 해당 캐시를 자유롭게 사용하세요!

### 자산 캐시 구조 예시[[assets-in-practice]]

실제로는 자산 캐시는 다음과 같은 트리 구조를 가질 것입니다:

```text
    assets/
    └── datasets/
    │   ├── SQuAD/
    │   │   ├── downloaded/
    │   │   ├── extracted/
    │   │   └── processed/
    │   ├── Helsinki-NLP--tatoeba_mt/
    │       ├── downloaded/
    │       ├── extracted/
    │       └── processed/
    └── transformers/
        ├── default/
        │   ├── something/
        ├── bert-base-cased/
        │   ├── default/
        │   └── training/
    hub/
    └── models--julien-c--EsperBERTo-small/
        ├── blobs/
        │   ├── (...)
        │   ├── (...)
        ├── refs/
        │   └── (...)
        └── [ 128]  snapshots/
            ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
            │   ├── (...)
            └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
                └── (...)
```

## 캐시 확인하기[[scan-your-cache]]

현재 캐시된 파일은 로컬 디렉토리에서 자동으로 삭제되지 않습니다. 브랜치의 새로운 수정 버전을 다운로드하면 이전 파일이 그대로 남아 향후 다시 사용할 수 있습니다. 따라서 디스크 공간을 많이 차지하는 리포지토리와 수정 버전을 파악하려면 캐시 디렉터리를 점검하는 것이 좋습니다. `huggingface_hub`은 이를 위해 CLI와 Python 유틸리티를 모두 제공합니다.

### 터미널에서 캐시 확인하기[[scan-cache-from-the-terminal]]

HF 캐시 상태를 살펴보는 가장 간단한 방법은 `hf cache ls` 명령을 사용하는 것입니다. 기본적으로 캐시에 저장된 리포지토리를 한눈에 보여줍니다.

```text
➜ hf cache ls
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

`--revisions` 옵션을 추가하면 각 스냅샷별 상세 목록을 확인할 수 있습니다. `size>1GB`, `accessed>30d`와 같이 사람이 읽기 쉬운 값을 사용하는 필터도 함께 적용할 수 있습니다.

```text
➜ hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
ID                                   REVISION            SIZE   LAST_MODIFIED REFS
------------------------------------ ------------------ ------- ------------- -------------------
model/bert-base-cased                6d1d7a1a2a6cf4c2    1.9G  2 years ago
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main

Found 2 repo(s) for a total of 2 revision(s) and 3.0G on disk.
```

머신 친화적인 출력이 필요하다면 `--format json` 또는 `--format csv`를 사용하고, 식별자만 받고 싶다면 `--quiet`를 사용할 수 있습니다. `--cache-dir` 옵션을 함께 지정하면 기본 위치가 아닌 다른 캐시 디렉터리를 살펴볼 수도 있습니다.

```text
➜ hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

#### 쉘 도구로 필터링하기[[grep-example]]

출력이 표 형식이므로 기존의 `grep` 같은 도구와도 잘 어울립니다. 아래 예시는 `t5-small` 관련 스냅샷만 찾는 방법입니다.

```text
➜ eval "hf cache ls --revisions" | grep "t5-small"
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main
model/t5-small                       8f3ad1c90fed7a62    820.1M 2 weeks ago   refs/pr/1
```

### 파이썬에서 캐시 스캔하기[[scan-cache-from-python]]

보다 고급 기능을 사용하려면, CLI 도구에서 호출되는 파이썬 유틸리티인 [`scan_cache_dir`]을 사용할 수 있습니다.

이를 사용하여 4가지 데이터 클래스를 중심으로 구조화된 자세한 보고서를 얻을 수 있습니다:

- [`HFCacheInfo`]: [`scan_cache_dir`]에 의해 반환되는 완전한 보고서
- [`CachedRepoInfo`]: 캐시된 리포지토리에 관한 정보
- [`CachedRevisionInfo`]: 리포지토리 내의 캐시된 수정 버전(예: "snapshot")에 관한 정보
- [`CachedFileInfo`]: 스냅샷 내의 캐시된 파일에 관한 정보

다음은 간단한 사용 예시입니다. 자세한 내용은 참조 문서를 참고하세요.

```py
>>> from huggingface_hub import scan_cache_dir

>>> hf_cache_info = scan_cache_dir()
HFCacheInfo(
    size_on_disk=3398085269,
    repos=frozenset({
        CachedRepoInfo(
            repo_id='t5-small',
            repo_type='model',
            repo_path=PosixPath(...),
            size_on_disk=970726914,
            nb_files=11,
            last_accessed=1662971707.3567169,
            last_modified=1662971107.3567169,
            revisions=frozenset({
                CachedRevisionInfo(
                    commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                    size_on_disk=970726339,
                    snapshot_path=PosixPath(...),
                    # 수정 버전 간에 blobs가 공유되기 때문에 `last_accessed`가 없습니다.
                    last_modified=1662971107.3567169,
                    files=frozenset({
                        CachedFileInfo(
                            file_name='config.json',
                            size_on_disk=1197
                            file_path=PosixPath(...),
                            blob_path=PosixPath(...),
                            blob_last_accessed=1662971707.3567169,
                            blob_last_modified=1662971107.3567169,
                        ),
                        CachedFileInfo(...),
                        ...
                    }),
                ),
                CachedRevisionInfo(...),
                ...
            }),
        ),
        CachedRepoInfo(...),
        ...
    }),
    warnings=[
        CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
        CorruptedCacheException(...),
        ...
    ],
)
```

## 캐시 정리하기[[clean-your-cache]]

캐시를 스캔하는 것은 흥미로울 수 있지만 실제로 해야 할 다음 작업은 일반적으로 드라이브의 일부 공간을 확보하기 위해 일부를 삭제하는 것입니다. 이는 `delete-cache` CLI 명령을 사용하여 가능합니다. 또한 캐시를 스캔할 때 반환되는 [`HFCacheInfo`] 객체에서 [`~HFCacheInfo.delete_revisions`] 헬퍼를 사용하여 프로그래밍 방식으로도 사용할 수 있습니다.

### 전략적으로 삭제하기[[delete-strategy]]


캐시를 삭제하려면 삭제할 수정 버전 목록을 전달해야 합니다. 이 도구는 이 목록을 기반으로 공간을 확보하기 위한 전략을 정의합니다. 이는 어떤 파일과 폴더가 삭제될지를 설명하는 [`DeleteCacheStrategy`] 객체를 반환합니다. [`DeleteCacheStrategy`]를 통해 사용 가능한 공간을 확보 할 수 있습니다. 삭제에 동의하면 삭제를 실행하여 삭제를 유효하게 만들어야 합니다. 불일치를 피하기 위해 전략 객체를 수동으로 편집할 수 없습니다.

수정 버전을 삭제하기 위한 전략은 다음과 같습니다:

- 수정 버전 심볼릭 링크가 있는 `snapshot` 폴더가 삭제됩니다.
- 삭제할 수정 버전에만 대상이 되는 blobs 파일도 삭제됩니다.
- 수정 버전이 1개 이상의 `refs`에 연결되어 있는 경우, 참조가 삭제됩니다.
- 리포지토리의 모든 수정 버전이 삭제되는 경우 전체 캐시된 리포지토리가 삭제됩니다.

> [!TIP]
> 수정 버전 해시는 모든 리포지토리를 통틀어 고유합니다. 따라서 `hf cache rm`은 리포지토리 식별자(`model/bert-base-uncased` 등)와 수정 버전 해시를 모두 인수로 받으며, 해시를 전달할 때는 리포지토리를 별도로 지정할 필요가 없습니다.

> [!WARNING]
> 캐시에서 수정 버전을 찾을 수 없는 경우 무시됩니다. 또한 삭제 중에 파일 또는 폴더를 찾을 수 없는 경우 경고가 기록되지만 오류가 발생하지 않습니다. [`DeleteCacheStrategy`] 객체에 포함된 다른 경로에 대해 삭제가 계속됩니다.

### 터미널에서 캐시 정리하기[[clean-cache-from-the-terminal]]

캐시에서 불필요한 데이터를 지우려면 `hf cache rm` 명령을 사용하세요. 리포지토리 식별자(예: `model/bert-base-uncased`)나 수정 버전 해시를 인수로 전달하면 됩니다.

```text
➜ hf cache rm model/bert-base-cased
About to delete 1 repo(s) totalling 1.9G.
  - model/bert-base-cased (entire repo)
Proceed with deletion? [y/N]: y
Deleted 1 repo(s) and 1 revision(s); freed 1.9G.
```

여러 리포지토리와 특정 수정 버전을 함께 지정할 수도 있습니다. `--dry-run` 옵션을 사용하면 실제 삭제 없이 결과를 미리 확인할 수 있고, 자동화된 스크립트에서는 `--yes`로 확인 단계를 건너뛸 수 있습니다.

```text
➜ hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```

기본 위치가 아닌 다른 캐시 디렉터리를 다루고 있다면 `--cache-dir` 옵션으로 경로를 지정하세요.

사용되지 않는(detached) 스냅샷만 한 번에 정리하려면 `hf cache prune`을 사용할 수 있습니다. 이 명령은 브랜치나 태그에서 참조하지 않는 수정 버전을 자동으로 선택합니다.

```text
➜ hf cache prune
About to delete 3 unreferenced revision(s) (2.4G total).
  - model/t5-small:
      1c610f6b [refs/pr/1] 820.1M
      d4ec9b72 [(detached)] 640.5M
  - dataset/google/fleurs:
      2b91c8dd [(detached)] 937.6M
Proceed? [y/N]: y
Deleted 3 unreferenced revision(s); freed 2.4G.
```

두 명령 모두 `--dry-run`, `--yes`, `--cache-dir` 옵션을 지원하므로 시뮬레이션, 자동화, 대체 캐시 경로 지정을 자유롭게 조합할 수 있습니다.

### 파이썬에서 캐시 정리하기[[clean-cache-from-python]]

더 유연하게 사용하려면, 프로그래밍 방식으로 [`~HFCacheInfo.delete_revisions`] 메소드를 사용할 수도 있습니다. 간단한 예제를 살펴보겠습니다. 자세한 내용은 참조 문서를 확인하세요.

```py
>>> from huggingface_hub import scan_cache_dir

>>> delete_strategy = scan_cache_dir().delete_revisions(
...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
...     "6c0e6080953db56375760c0471a8c5f2929baf11",
... )
>>> print("Will free " + delete_strategy.expected_freed_size_str)
Will free 8.6G

>>> delete_strategy.execute()
Cache deletion done. Saved 8.6G.
```
