<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interact with the Hub through the Filesystem API

In addition to the [`HfApi`], the `huggingface_hub` library provides [`HfFileSystem`], a pythonic [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) file interface to the Hugging Face Hub. The [`HfFileSystem`] builds of top of the [`HfApi`] and offers typical filesystem style operations like `cp`, `mv`, `ls`, `du`, `glob`, `get_file`, and `put_file`.

## Usage

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem()

>>> # List all files in a directory
>>> fs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # List all ".csv" files in a repo
>>> fs.glob("datasets/my-username/my-dataset-repo/**/*.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Read a remote file
>>> with fs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # Read the content of a remote file as a string
>>> train_data = fs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # Write a remote file
>>> with fs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

The optional `revision` argument can be passed to run an operation from a specific commit such as a branch, tag name, or a commit hash.

Unlike Python's built-in `open`, `fsspec`'s `open` defaults to binary mode, `"rb"`. This means you must explicitly set mode as `"r"` for reading and `"w"` for writing in text mode. Appending to a file (modes `"a"` and `"ab"`) is not supported yet.

## Integrations

The [`HfFileSystem`] can be used with any library that integrates `fsspec`, provided the URL follows the scheme:

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/hf_urls.png"/>
</div>

The `repo_type_prefix` is `datasets/` for datasets, `spaces/` for spaces, and models don't need a prefix in the URL.

Some interesting integrations where [`HfFileSystem`] simplifies interacting with the Hub are listed below:

* Reading/writing a [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) DataFrame from/to a Hub repository:

  ```python
  >>> import pandas as pd

  >>> # Read a remote CSV file into a dataframe
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")

  >>> # Write a dataframe to a remote CSV file
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  ```

The same workflow can also be used for [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) and [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html) DataFrames.

* Querying (remote) Hub files with [DuckDB](https://duckdb.org/docs/guides/python/filesystems):

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # Query a remote file and get the result back as a dataframe
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* Using the Hub as an array store with [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec):

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # Write an array to a repo
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # Read an array from a repo
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```

## Authentication

In many cases, you must be logged in with a Hugging Face account to interact with the Hub. Refer to the [Authentication](../quick-start#authentication) section of the documentation to learn more about authentication methods on the Hub.

It is also possible to log in programmatically by passing your `token` as an argument to [`HfFileSystem`]:

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem(token=token)
```

If you log in this way, be careful not to accidentally leak the token when sharing your source code!
