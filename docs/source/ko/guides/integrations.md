<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 어떤 머신 러닝 프레임워크든 Hub와 통합[[integrate-any-ml-framework-with-the-hub]]
Hugging Face Hub는 커뮤니티와 모델을 공유하는 것을 쉽게 만들어줍니다. 이는 오픈소스 생태계의 [수십 가지 라이브러리](https://glorious-goldfish-v557q9qvxv7366v9.github.dev/)를 지원합니다. 저희는 항상 협업적인 머신 러닝을 전진시키기 위해 이 지원을 확대하기 위해 노력하고 있습니다. `huggingface_hub` 라이브러리는 어떤 Python 스크립트든지 쉽게 파일을 업로드하고 로드할 수 있도록 하는 데 중요한 역할을 합니다.

라이브러리를 Hub와 통합하는 주요 네 가지 방법이 있습니다:

1. **Hub에 업로드하기**: 모델을 Hub에 업로드하는 메서드를 구현합니다. 이에는 모델 가중치뿐만 아니라 [모델 카드](https://huggingface.co/docs/huggingface_hub/how-to-model-cards) 및 모델 실행에 필요한 다른 관련 정보나 데이터(예: 훈련 로그)가 포함됩니다. 이 방법은 일반적으로 `push_to_hub()`라고 합니다.
2. **Hub에서 다운로드하기**: Hub에서 모델을 가져오는 메서드를 구현합니다. 이 메서드는 모델 구성/가중치를 다운로드하고 모델을 가져와야 합니다. 이 방법은 일반적으로 `from_pretrained` 또는 `load_from_hub()`라고 합니다.
3. **추론 API**: 라이브러리에서 지원하는 모델에 대해 무료로 추론을 실행하기 위해 서버를 사용합니다.
4. **위젯**: Hub의 모델 랜딩 페이지에 위젯을 표시합니다. 이를 통해 사용자들은 브라우저에서 빠르게 모델을 시도할 수 있습니다.

이 가이드에서는 앞의 두 가지 주제에 중점을 둘 것입니다. 두 가지 주요 통합 방법을 소개하고 각각의 장단점을 설명할 것입니다. 이를 요약하여 두 가지 중 어떤 것을 선택할지에 대한 도움이 될 것입니다. 이는 단지 가이드라는 것을 명심하고 상황에 맞게 적응시킬 수 있는 가이드라는 점을 유념하십시오.

추론 및 위젯에 관심이 있는 경우 [이 가이드](https://huggingface.co/docs/hub/models-adding-libraries#set-up-the-inference-api)를 참조할 수 있습니다. 양쪽 모두에서 라이브러리를 Hub와 통합하고 [문서](https://huggingface.co/docs/hub/models-libraries)에 목록에 게시하고자 하는 경우에는 언제든지 저희에게 연락하실 수 있습니다.

## 유연한 접근 방식: 도우미(helper)[[a-flexible-approach-helpers]]

라이브러리를 Hub에 통합하는 첫 번째 접근 방법은 실제로 `push_to_hub` 및 `from_pretrained` 메서드를 직접 구현하는 것입니다. 이를 통해 업로드/다운로드할 파일 및 입력을 처리하는 방법에 대한 완전한 유연성을 제공받을 수 있습니다. 이를 위해 [파일 업로드](./upload) 및 [파일 다운로드](./download) 가이드를 참조하여 자세히 알아볼 수 있습니다. 예를 들어 FastAI 통합이 구현된 방법을 보면 됩니다 ([`push_to_hub_fastai`] 및 [`from_pretrained_fastai`]를 참조).

라이브러리마다 구현 방식은 다를 수 있지만, 워크플로우는 일반적으로 비슷합니다.

### from_pretrained[[frompretrained]]

일반적으로 `from_pretrained` 메서드는 다음과 같은 형태를 가집니다:

```python
def from_pretrained(model_id: str) -> MyModelClass:
   # Hub로부터 모델을 다운로드
   cached_model = hf_hub_download(
      repo_id=repo_id,
      filename="model.pkl",
      library_name="fastai",
      library_version=get_fastai_version(),
   )

   # 모델 가져오기
    return load_model(cached_model)
```

### push_to_hub[[pushtohub]]

`push_to_hub` 메서드는 종종 레포지토리 생성, 모델 카드 생성 및 가중치 저장을 처리하기 위해 조금 더 복잡한 접근 방식이 필요합니다. 일반적으로 모든 이러한 파일을 임시 폴더에 저장한 다음 업로드하고 나중에 삭제하는 방식이 흔히 사용됩니다.

```python
def push_to_hub(model: MyModelClass, repo_name: str) -> None:
   api = HfApi()

   # 해당 리포지토리가 아직 없다면 리포지토리를 생성하고 관련된 리포지토리 ID를 가져옵니다.
   repo_id = api.create_repo(repo_name, exist_ok=True)

   # 모든 파일을 임시 디렉토리에 저장하고 이를 단일 커밋으로 푸시합니다.
   with TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      # 가중치 저장
      save_model(model, tmpdir / "model.safetensors")

      # model card 생성
      card = generate_model_card(model)
      (tmpdir / "README.md").write_text(card)

      # 로그 저장
      # 설정 저장
      # 평가 지표를 저장
      # ...

      # Hub에 푸시
      return api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
```


물론 이는 단순한 예시에 불과합니다. 더 복잡한 조작(원격 파일 삭제, 가중치를 실시간으로 업로드, 로컬로 가중치를 유지 등)에 관심이 있다면 [파일 업로드](./upload) 가이드를 참조해 주세요.

### 제한[[limitations]]

유연성을 가지고 있지만, 이 방식은 유지보수 측면에서 일부 단점을 가지고 있습니다. Hugging Face 사용자들은 `huggingface_hub`와 함께 작업할 때 추가 기능에 익숙합니다. 예를 들어, Hub에서 파일을 로드할 때 다음과 같은 매개변수를 제공하는 것이 일반적입니다:

- `token`: 개인 리포지토리에서 다운로드하기 위한 토큰
- `revision`: 특정 브랜치에서 다운로드하기 위한 리비전
- `cache_dir`: 특정 디렉터리에 파일을 캐시하기 위한 디렉터리
- `force_download`/`resume_download`/`local_files_only`: 캐시를 재사용할 것인지 여부를 결정하는 매개변수
- `proxies`: HTTP 세션 구성

모델을 푸시할 때는 유사한 매개변수가 지원됩니다:
- `commit_message`: 사용자 정의 커밋 메시지
- `private`: 개인 리포지토리를 만들어야 할 경우
- `create_pr`: `main`에 푸시하는 대신 PR을 만드는 경우
- `branch`: `main` 브랜치 대신 브랜치에 푸시하는 경우
- `allow_patterns/ignore_patterns`: 업로드할 파일을 필터링하는 매개변수
- `token`
- ...

이러한 매개변수는 위에서 본 구현에 추가하여 `huggingface_hub` 메서드로 전달할 수 있습니다. 그러나 매개변수가 변경되거나 새로운 기능이 추가되는 경우에는 패키지를 업데이트해야 합니다. 이러한 매개변수를 지원하는 것은 유지 관리할 문서가 더 많아진다는 것을 의미합니다. 이러한 제한 사항을 어떻게 완화할 수 있는지 보려면 다음 섹션인 **클래스 상속**으로 이동해 보겠습니다.

## 더욱 복잡한 접근법: 클래스 상속[[a-more-complex-approach-class-inheritance]]


위에서 보았듯이 Hub와 통합하기 위해 라이브러리에 포함해야 할 주요 메서드는 파일을 업로드하는 (`push_to_hub`) 및 파일을 다운로드하는 (`from_pretrained`)입니다. 이러한 메서드를 직접 구현할 수 있지만, 이에는 몇 가지 주의할 점이 있습니다. 이를 해결하기 위해 `huggingface_hub`은 클래스 상속을 사용하는 도구를 제공합니다. 이 도구가 어떻게 작동하는지 살펴보겠습니다!

많은 경우에 라이브러리는 이미 Python 클래스를 사용하여 모델을 구현합니다. 이 클래스에는 모델의 속성 및 로드, 실행, 훈련 및 평가하는 메서드가 포함되어 있습니다. 접근 방식은 [믹스인](https://stackoverflow.com/a/547714)을 사용하여 이 클래스를 확장하여 업로드 및 다운로드 기능을 포함하는 것입니다. 믹스인(Mixin)은 기존 클래스에 여러 상속을 통해 특정 기능을 확장하기 위해 설계된 클래스입니다. `huggingface_hub`은 자체 믹스인인 [`ModelHubMixin`]을 제공합니다. 이 중요한 점은 그 동작을 이해하고 사용자 정의하는 것입니다.

[`ModelHubMixin`] 클래스는 세 개의 *공개* 메서드(`push_to_hub`, `save_pretrained`, `from_pretrained`)를 구현합니다. 이 메서드들은 사용자가 라이브러리를 사용하여 모델을 로드/저장할 때 호출하는 메서드입니다. 또한 [`ModelHubMixin`]은 두 개의 *비공개* 메서드(`_save_pretrained` 및 `_from_pretrained`)를 정의합니다. 이 메서드들을 구현해야 합니다. 라이브러리를 통합하려면:

1. 모델 클래스를 [`ModelHubMixin`]에서 상속합니다.
2. 비공개 메서드를 구현합니다:
   - [`~ModelHubMixin._save_pretrained`]: 디렉터리 경로를 입력으로 받아 모델을 해당 디렉터리에 저장하는 메서드입니다. 이 메서드에는 모델 카드, 모델 가중치, 구성 파일, 훈련 로그 및 그림 등 해당 모델에 대한 모든 관련 정보를 덤프하기 위한 로직을 작성해야 합니다. [모델 카드](https://huggingface.co/docs/hub/model-cards)는 모델을 설명하는 데 특히 중요합니다. 더 자세한 내용은 [구현 가이드](./model-cards)를 확인하세요.
   - [`~ModelHubMixin._from_pretrained`]: `model_id`를 입력으로 받아 인스턴스화된 모델을 반환하는 **클래스 메서드**입니다. 이 메서드는 관련 파일을 다운로드하고 로드해야 합니다.
3. 완료했습니다!

[`ModelHubMixin`]의 장점은 파일의 직렬화/로드에만 신경을 쓰면 되기 때문에 즉시 사용할 수 있다는 것입니다. 레포지토리 생성, 커밋, PR 또는 리비전과 같은 사항에 대해 걱정할 필요가 없습니다. 이 모든 것은 믹스인에 의해 처리되며 사용자에게 제공됩니다. 믹스인은 또한 공개 메서드가 잘 문서화되고 타입이 주석이 달려있는지를 보장합니다.

보너스로, [`ModelHubMixin`]은 모델 구성을 자동으로 처리해 줍니다. 만약 당신의 `__init__` 메서드가 `config` 입력을 기대한다면, `save_pretrained`를 호출할 때 자동으로 레포에 저장되고 `load_pretrained`에 의해 올바르게 다시 로드될 것입니다. 더불어, `config` 입력 매개변수가 dataclass 타입으로 주석 처리되어 있다면 (예: `config: Optional[MyConfigClass] = None`), 그렇게 하면 `config` 값이 올바르게 역직렬화됩니다. 마지막으로, 초기화할 때 전달된 모든 jsonable 값은 구성 파일에 저장됩니다. 이는 `config` 입력을 기대하지 않더라도 이를 활용할 수 있다는 것을 의미합니다. 모델 리포지토리에 `config.json` 파일이 있으면 Hub에서 자동으로 분석을 활성화시킵니다 (예: "다운로드" 횟수).

마지막으로, [`ModelHubMixin`]은 모델 카드 생성을 처리해줍니다. [`ModelHubMixin`]을 상속받을 때 `library_name`, `tags`, `repo_url`, `docs_url`과 같은 메타데이터를 정의할 수 있습니다. 이러한 필드는 클래스를 사용하는 모든 모델의 모델 카드를 채우는 데 재사용됩니다. 이는 Hub에서 라이브러리를 사용하는 모든 모델을 쉽게 검색할 수 있도록 만들고, Hub에 착륙하는 사용자에게 일부 리소스 링크를 제공하는 데 매우 유용합니다. 만약 모델 카드 템플릿을 확장하고 싶다면, [`~ModelHubMixin.generate_model_card`] 메서드를 재정의할 수 있습니다.

### 자세한 예시: PyTorch[[a-concrete-example-pytorch]]

위에서 언급한 내용의 좋은 예시는 [`PyTorchModelHubMixin`]입니다. 이것은 PyTorch 프레임워크를 위한 저희의 통합입니다. 이것은 즉시 사용할 수 있는 통합입니다.

#### 어떻게 사용하나요?[[how-to-use-it]]

다음은 Hub에서 PyTorch 모델을 로드/저장하는 방법입니다:

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin


# PyTorch 모델을 여러분이 흔히 사용하는 방식과 완전히 동일하게 정의하세요.
>>> class MyModel(
...         nn.Module,
...         PyTorchModelHubMixin, # 다중 상속
...         library_name="keras-nlp",
...         tags=["keras"],
...         repo_url="https://github.com/keras-team/keras-nlp",
...         docs_url="https://keras.io/keras_nlp/",
...         # ^ 모델 카드를 생성하는 데 선택적인 메타데이터입니다.
...     ):
...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
...         super().__init__()
...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
...         self.linear = nn.Linear(output_size, vocab_size)

...     def forward(self, x):
...         return self.linear(x + self.param)

# 1. 모델 생성
>>> model = MyModel(hidden_size=128)

# 설정은 입력 및 기본값을 기반으로 자동으로 생성됩니다.
>>> model.param.shape[0]
128

# 2. (선택사항) 모델을 로컬 디렉터리에 저장합니다.
>>> model.save_pretrained("path/to/my-awesome-model")

# 3. 모델 가중치를 Hub에 푸시합니다.
>>> model.push_to_hub("my-awesome-model")

# 4. Hub로부터 모델을 초기화합니다. => 이때 설정은 보존됩니다.
>>> model = MyModel.from_pretrained("username/my-awesome-model")
>>> model.param.shape[0]
128

# 모델 카드가 올바르게 작성되었습니다.
>>> from huggingface_hub import ModelCard
>>> card = ModelCard.load("username/my-awesome-model")
>>> card.data.tags
["keras", "pytorch_model_hub_mixin", "model_hub_mixin"]
>>> card.data.library_name
"keras-nlp"
```

#### 구현[[implementation]]

실제 구현은 매우 간단합니다. 전체 구현은 [여기](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py)에서 찾을 수 있습니다.

1. 클래스를 `ModelHubMixin`으로부터 상속하세요:

```python
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)
```

2. `_save_pretrained` 메서드를 구현하세요:

```py
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)

    def _save_pretrained(self, save_directory: Path) -> None:
        """PyTorch 모델의 가중치를 로컬 디렉터리에 저장합니다."""
        save_model_as_safetensor(self.module, str(save_directory / SAFETENSORS_SINGLE_FILE))

```

3. `_from_pretrained` 메서드를 구현하세요:

```python
class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   @classmethod # 반드시 클래스 메서드여야 합니다!
   def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: str,
      cache_dir: str,
      force_download: bool,
      proxies: Optional[Dict],
      resume_download: bool,
      local_files_only: bool,
      token: Union[str, bool, None],
      map_location: str = "cpu", # 추가 인자
      strict: bool = False, # 추가 인자
      **model_kwargs,
   ):
      """PyTorch의 사전 학습된 가중치를 로드하고 로드된 모델을 반환합니다."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)

         model_file = hf_hub_download(
            repo_id=model_id,
            filename=SAFETENSORS_SINGLE_FILE,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
            )
         return cls._load_as_safetensor(model, model_file, map_location, strict)
```

이게 전부입니다! 이제 라이브러리를 통해 Hub로부터 파일을 업로드하고 다운로드할 수 있습니다.

## 빠른 비교[[quick-comparison]]

두 가지 접근 방법에 대한 장단점을 간단히 정리해보겠습니다. 아래 표는 단순히 예시일 뿐입니다. 여러분의 프레임워크에는 고려해야 할 특정 사항이 있을 수 있습니다. 이 가이드는 통합을 다루는 아이디어와 지침을 제공하기 위한 것입니다. 언제든지 궁금한 점이 있으면 문의해 주세요!

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->
| Integration | Using helpers | Using [`ModelHubMixin`] |
|:---:|:---:|:---:|
| User experience | `model = load_from_hub(...)`<br>`push_to_hub(model, ...)` | `model = MyModel.from_pretrained(...)`<br>`model.push_to_hub(...)` |
| Flexibility | Very flexible.<br>You fully control the implementation. | Less flexible.<br>Your framework must have a model class. |
| Maintenance | More maintenance to add support for configuration, and new features. Might also require fixing issues reported by users. | Less maintenance as most of the interactions with the Hub are implemented in `huggingface_hub`. |
| Documentation / Type annotation | To be written manually. | Partially handled by `huggingface_hub`. |
| Download counter | To be handled manually. | Enabled by default if class has a `config` attribute. |
| Model card | To be handled manually | Generated by default with library_name, tags, etc. |
