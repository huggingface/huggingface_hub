<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Hugging Face Hub에서 파일 시스템 API를 통해 상호작용하기[[interact-with-the-hub-through-the-filesystem-api]]

`huggingface_hub` 라이브러리는 [`HfApi`] 외에도 Hugging Face Hub에 대한 파이써닉한 [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) 파일 인터페이스인 [`HfFileSystem`]을 제공합니다. [`HfFileSystem`]은 [`HfApi`]을 기반으로 구축되며, `cp`, `mv`, `ls`, `du`, `glob`, `get_file` 및 `put_file`과 같은 일반적인 파일 시스템 스타일 작업을 제공합니다.

## 사용법[[usage]]

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem()

>>> # 디렉터리의 모든 파일 나열하기
>>> fs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # 저장소(repo)에서 ".csv" 파일 모두 나열하기
>>> fs.glob("datasets/my-username/my-dataset-repo/**.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # 원격 파일 읽기
>>> with fs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # 문자열로 원격 파일의 내용 읽기
>>> train_data = fs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # 원격 파일 쓰기
>>> with fs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

선택적 `revision` 인수를 전달하여 브랜치, 태그 이름 또는 커밋 해시와 같은 특정 커밋에서 작업을 실행할 수 있습니다.

파이썬에 내장된 `open`과 달리 `fsspec`의 `open`은 바이너리 모드 `"rb"`로 기본 설정됩니다. 이것은 텍스트 모드에서 읽기 위해 `"r"`, 쓰기 위해 `"w"`로 모드를 명시적으로 설정해야 함을 의미합니다. 파일에 추가하기(모드 `"a"` 및 `"ab"`)는 아직 지원되지 않습니다.

## 통합[[integrations]]

[`HfFileSystem`]은 URL이 다음 구문을 따르는 경우 `fsspec`을 통합하는 모든 라이브러리에서 사용할 수 있습니다.

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

여기서 `repo_type_prefix`는 Datasets의 경우 `datasets/`, Spaces의 경우 `spaces/`이며, 모델에는 URL에 접두사가 필요하지 않습니다.

[`HfFileSystem`]이 Hub와의 상호작용을 단순화하는 몇 가지 흥미로운 통합 사례는 다음과 같습니다:

* Hub 저장소에서 [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) DataFrame 읽기/쓰기:

  ```python
  >>> import pandas as pd

  >>> # 원격 CSV 파일을 데이터프레임으로 읽기
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")

  >>> # 데이터프레임을 원격 CSV 파일로 쓰기
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  ```

동일한 워크플로우를 [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) 및 [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html) DataFrame에도 사용할 수 있습니다.

* [DuckDB](https://duckdb.org/docs/guides/python/filesystems)를 사용하여 (원격) Hub 파일 쿼리:

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # 원격 파일을 쿼리하고 결과를 데이터프레임으로 가져오기
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec)를 사용하여 Hub를 배열 저장소로 사용:

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # 저장소(repo)에 배열 쓰기
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # 저장소(repo)에서 배열 읽기
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```

## 인증[[authentication]]

대부분의 경우 Hub와 상호작용하려면 Hugging Face 계정에 로그인해야 합니다. Hub에서 인증 방법에 대해 자세히 알아보려면 문서의 [인증](../quick-start#authentication) 섹션을 참조하세요.

또한 [`HfFileSystem`]에 `token`을 인수로 전달하여 프로그래밍 방식으로 로그인할 수 있습니다:

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem(token=token)
```

이렇게 로그인하는 경우 소스 코드를 공유할 때 토큰이 실수로 누출되지 않도록 주의해야 합니다!
