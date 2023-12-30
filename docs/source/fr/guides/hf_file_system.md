<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Intéragissez avec le Hub à partir de l'API Filesystem

En plus d'[`HfApi`], la librairie `huggingface_hub` fournit [`HfFileSystem`], une interface vers le Hub Hugging Face, basée sur Python, et [compatible fsspec](https://filesystem-spec.readthedocs.io/en/latest/). [`HfFileSystem`] fournit les opérations classiques des filesystem telles que
`cp`, `mv`, `ls`, `du`, `glob`, `get_file`, et `put_file`.

## Utilisation

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem()

>>> # Liste tous les fichiers d'un chemin
>>> fs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Liste tous les fichiers ".csv" d'un dépôt
>>> fs.glob("datasets/my-username/my-dataset-repo/**.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Lis un fichier distant 
>>> with fs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # Lis le contenu d'un fichier distant en renvoyant un string
>>> train_data = fs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # Lis un fichier distant
>>> with fs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

L'argument optionnel `revision` peut être passé pour exécuter une opération sur un commit spécifique en précisant la branche, le tag, ou un hash de commit.

A la différence des fonction native de Python `open`, la fonction `open` de `fsspec` est en mode binaire par défaut, `"rb"`. Ceci signifie que vous devez explicitement définir le mode à `"r"` pour lire et `"w"` pour écrire en mode texte. Les modes `"a"` et `"ab"` ne sont pas encore supportés.

## Intégrations

[`HfFileSystem`] peut être utilisé avec toutes les librairies qui intègrent `fsspec`, tant que l'URL a le schéma suivant:

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

Le `repo_type_prefix` vaut `datasets/` pour les datasets, `spaces/` pour les espaces, et les modèles n'ont pas besoin de préfixe dans l'URL.

Ci-dessous quelques intégrations intéressantes où [`HfFileSystem`] simplifie l'intéraction avec le Hub:

* Lire/modifier un dataframe [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) depuis/vers un dépôt du Hub:

  ```python
  >>> import pandas as pd

  >>> # Lis un fichier CSV distant en renvoyant un dataframe
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")

  >>> # Enregistre un dataframe vers un fichier CSV distant
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  ```

Un workflow similaire peut-être utilisé pour les dataframes [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) et [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html)

* Requête afin d'obtenir des fichiers du Hub (distants) avec [DuckDB](https://duckdb.org/docs/guides/python/filesystems): 

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # Requête pour obtenir un fichier distant et récupérer les résultats sous forme de dataframe
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* Utilisation du Hub pour stocker des tableau avec [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec):

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # Écriture d'un tableau vers un dépôt
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # Lecture d'un tableau depuis un dépôt
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```

## Authentification

Souvent, vous devrez être connecté avec un compte Hugging Face pour intéragir avec le Hub. Consultez la section [connexion](../quick-start#login) de la documentation pour en apprendre plus sur les méthodes d'authentifications sur le Hub.

Il est aussi possible de se connecter par le code en passant l'agument `token` à [`HfFileSystem`]:

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem(token=token)
```

Si vous vous connectez de cette manière, faites attention à ne pas accidentellement révéler votre token en cas de partage du code source!
