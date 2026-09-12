<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Interagir avec le Hub via l'API Filesystem

En plus de [`HfApi`], la bibliothèque `huggingface_hub` fournit [`HfFileSystem`], une interface de fichiers  [compatible fsspec](https://filesystem-spec.readthedocs.io/en/latest/) vers le Hugging Face Hub. [`HfFileSystem`] s'appuie sur [`HfApi`] et offre des opérations typiques de systèmes de fichiers comme `cp`, `mv`, `ls`, `du`, `glob`, `get_file` et `put_file`.

> [!WARNING]
> [`HfFileSystem`] fournit une compatibilité fsspec, ce qui est utile pour les bibliothèques qui le requièrent (par exemple, lire
> des datasets Hugging Face directement avec `pandas`). Cependant, cela introduit une surcharge supplémentaire due à cette
> couche de compatibilité. Pour de meilleures performances et fiabilité, il est recommandé d'utiliser les méthodes [`HfApi`] lorsque c'est possible.

## Utilisation

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem()

>>> # Lister tous les fichiers dans un répertoire
>>> fs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Lister tous les fichiers ".csv" dans un dépôt
>>> fs.glob("datasets/my-username/my-dataset-repo/**/*.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Lire un fichier distant
>>> with fs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # Lire le contenu d'un fichier distant comme une chaîne
>>> train_data = fs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # Écrire un fichier distant
>>> with fs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

L'argument optionnel `revision` peut être passé pour exécuter une opération depuis un commit spécifique tel qu'une branche, un nom de tag ou un hash de commit.

Contrairement au `open` intégré de Python, le `open` de `fsspec` utilise par défaut le mode binaire, `"rb"`. Cela signifie que vous devez explicitement définir le mode comme `"r"` pour lire et `"w"` pour écrire en mode texte. L'ajout à un fichier (modes `"a"` et `"ab"`) n'est pas encore supporté.

## Intégrations

[`HfFileSystem`] peut être utilisé avec n'importe quelle bibliothèque qui intègre `fsspec`, à condition que l'URL suive le schéma :

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/hf_urls.png"/>
</div>

Le `repo_type_prefix` est `datasets/` pour les datasets, `spaces/` pour les spaces, et les modèles n'ont pas besoin de préfixe dans l'URL.

Voici quelques intégrations intéressantes où [`HfFileSystem`] simplifie l'interaction avec le Hub :

* Lire/écrire un DataFrame [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) depuis/vers un dépôt Hub :

  ```python
  >>> import pandas as pd

  >>> # Lire un fichier CSV distant dans un dataframe
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")

  >>> # Écrire un dataframe dans un fichier CSV distant
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  ```

Le même workflow peut également être utilisé pour les DataFrames [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) et [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html).

* Interroger des fichiers Hub (distants) avec [DuckDB](https://duckdb.org/docs/guides/python/filesystems) :

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # Interroger un fichier distant et obtenir le résultat sous forme de dataframe
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* Utiliser le Hub comme stockage de tableaux avec [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec) :

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # Écrire un tableau dans un dépôt
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # Lire un tableau depuis un dépôt
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```

## Authentification

Dans de nombreux cas, vous devez être connecté avec un compte Hugging Face pour interagir avec le Hub. Consultez la section [Authentification](../quick-start#authentication) de la documentation pour en savoir plus sur les méthodes d'authentification sur le Hub.

Il est également possible de se connecter en passant votre `token` comme argument à [`HfFileSystem`] :

```python
>>> from huggingface_hub import HfFileSystem
>>> fs = HfFileSystem(token=token)
```

Si vous vous connectez de cette façon, veillez à ne pas divulguer accidentellement le token lors du partage de votre code source !
