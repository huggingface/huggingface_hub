<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Faites des recherches dans le Hub

Dans ce tutoriel, vous apprendrez à chercher des modèles, des datasets et des spaces du Hub en utilisant `huggingface_hub`.

## Comment lister les dépôts ?

La librairie `huggingface_hub` inclus un client HTTP [`HfApi`] pour intéragir avec le Hub.
Ce client peut, entre autres, lister les modèles, les dataset et les spaces enregistrés sur le Hub:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

La sortie de [`list_models`] est un itérateur sur les modèles stockés dans le Hub.

De la même manière, vous pouvez utiliser [`list_datasets`] pour lister les datasets et [`list_spaces`] pour lister les spaces.

## Comment filtrer des dépôts ?

Lister les dépôts est très utile, mais vous aurez surement besoin de filtrer votre recherche.
Les helpers ont plusieurs attributs tels que:
- `filter`
- `author`
- `search`
- ...

Deux de ces paramètres sont assez intuitifs (`author` et `search`) mais qu'en est il de `filter`?
`filter` prend en entrée un objet [`ModelFilter`] (ou [`DatasetFilter`]). Vous pouvez l'instancier
en précisang quels modèles vous voulez filtrer. 

Regaardons comment nous pouvons avoir tous les modèles sur le Hub qui font de la classification
d'images, qui ont été entrainé sur le dataset imagenet et qui utilisent PyTorch. On peut le
faire en utilisant un seul [`ModelFilter`]. Les attributs sont combinés comme des "ET" logiques:

```py
models = hf_api.list_models(
    filter=ModelFilter(
		task="image-classification",
		library="pytorch",
		trained_dataset="imagenet"
	)
)
```

Lors du filtrage, vous pouvez aussi trier les modèles en prendre uniquement les premiers
résultats. L'exemple suivant récupère les 5 datasets les plus téléchargés du Hub:

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



Pour explorer tous les filtres disponibles sur le HUb, consultez les pages [modèles](https://huggingface.co/models) et [datasets](https://huggingface.co/datasets) dans votre navigateur, cherchez des paramètres et regardez les valeurs dans l'URL.

