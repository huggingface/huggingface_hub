<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interagieren mit dem Hub über die Filesystem API

Zusätzlich zur [`HfApi`] bietet die `huggingface_hub` Bibliothek [`HfFileSystem`], eine pythonische, [fsspec-kompatible](https://filesystem-spec.readthedocs.io/en/latest/) Dateischnittstelle zum Hugging Face Hub. Das [`HfFileSystem`] basiert auf der [`HfApi`] und bietet typische Dateisystemoperationen wie `cp`, `mv`, `ls`, `du`, `glob`, `get_file`, und `put_file`.

## Verwendung

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem()

>>> # Alle Dateien in einem Verzeichnis auflisten
>>> fs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Alle ".csv"-Dateien in einem Repo auflisten
>>> fs.glob("datasets/my-username/my-dataset-repo/**.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Eine entfernte Datei lesen
>>> with fs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # Den Inhalt einer entfernten Datei als Zeichenkette / String lesen
>>> train_data = fs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # Eine entfernte Datei schreiben
>>> with fs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

Das optionale Argument `revision` kann übergeben werden, um eine Operation von einem spezifischen Commit auszuführen, wie z.B. einem Branch, Tag-Namen oder einem Commit-Hash.

Anders als bei Pythons eingebautem `open`, ist der Standardmodus von `fsspec`'s `open` binär, `"rb"`. Das bedeutet, dass Sie den Modus explizit auf `"r"` zum Lesen und `"w"` zum Schreiben im Textmodus setzen müssen. Das Anhängen an eine Datei (Modi `"a"` und `"ab"`) wird noch nicht unterstützt.

## Integrationen

Das [`HfFileSystem`] kann mit jeder Bibliothek verwendet werden, die `fsspec` integriert, vorausgesetzt die URL folgt dem Schema:

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<pfad/im/repo>
```

Der `repo_type_prefix` ist `datasets/` für Datensätze, `spaces/` für Spaces, und Modelle benötigen kein Präfix in der URL.

Einige interessante Integrationen, bei denen [`HfFileSystem`] die Interaktion mit dem Hub vereinfacht, sind unten aufgeführt:

* Lesen/Schreiben eines [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) DataFrame aus/in ein Hub-Repository:

  ```python
  >>> import pandas as pd

  >>> # Eine entfernte CSV-Datei in einen DataFrame lesen
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")

  >>> # Einen DataFrame in eine entfernte CSV-Datei schreiben
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  ```

  Der gleiche Arbeitsablauf kann auch für  [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) und [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html) verwendet werden.

* Abfrage von (entfernten) Hub-Dateien mit  [DuckDB](https://duckdb.org/docs/guides/python/filesystems):

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # Eine entfernte Datei abfragen und das Ergebnis als DataFrame zurückbekommen
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* Verwendung des Hub als Array-Speicher mit [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec):

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # Ein Array in ein Repo schreiben
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # Ein Array aus einem Repo lesen
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```

## Authentifizierung

In vielen Fällen müssen Sie mit einem Hugging Face-Konto angemeldet sein, um mit dem Hub zu interagieren. Lesen Sie den [Login](../quick-start#login)-Abschnitt der Dokumentation, um mehr über Authentifizierungsmethoden auf dem Hub zu erfahren.

Es ist auch möglich, sich programmatisch anzumelden, indem Sie Ihr `token` als Argument an [`HfFileSystem`] übergeben:


```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem(token=token)
```

Wenn Sie sich auf diese Weise anmelden, seien Sie vorsichtig, das Token nicht versehentlich zu veröffentlichen, wenn Sie Ihren Quellcode teilen!
