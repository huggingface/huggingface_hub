<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Rechercher dans le Hub

Dans ce tutoriel, vous apprendrez comment rechercher des modèles, jeux de données (datasets) et spaces sur le Hub en utilisant la bibliothèque `huggingface_hub`.

## Comment lister les dépôts ?

La bibliothèque `huggingface_hub` inclut un client HTTP [`HfApi`] pour interagir avec le Hub.
En utilisant ce client, vous pouvez lister les modèles, des jeux de données et spaces stockés sur le Hub :

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

La sortie de [`list_models`] est une liste des modèles stockés sur le Hub.

De même, vous pouvez utiliser [`list_datasets`] pour lister les jeux de données et [`list_spaces`] pour lister les Spaces.

## Comment filtrer les dépôts ?

Lister les dépôts est bien, mais maintenant vous pourriez vouloir filtrer votre recherche.
Les helpers de liste ont plusieurs attributs comme :
- `filter`
- `author`
- `search`
- ...

Voyons un exemple pour obtenir tous les modèles sur le Hub qui font de la classification d'images, qui ont été entraînés sur le dataset "imagenet" et qui fonctionnent avec PyTorch.

```py
models = hf_api.list_models(filter=["image-classification", "pytorch", "imagenet"])
```

Lors du filtrage, vous pouvez également trier les modèles et ne prendre que les meilleurs résultats. Par exemple,
l'exemple suivant récupère les 5 jeux de données les plus téléchargés sur le Hub :

```py
>>> list(list_datasets(sort="downloads", direction=-1, limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```

Pour explorer les filtres disponibles sur le Hub, visitez les pages [models](https://huggingface.co/models) et [datasets](https://huggingface.co/datasets)
dans votre navigateur, recherchez des paramètres et regardez les valeurs dans l'URL. (Paramètre de méthode GET après le `?=` ou `&=`).
