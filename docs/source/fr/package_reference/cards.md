<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Cartes de dépôts

La librairie `huggingface_hub` fournit une interface Python pour créer, partager et mettre à jour les
cartes de modèles ou de dataset. Consultez la [page de documentation dédiée](https://huggingface.co/docs/hub/models-cards)
Pour une vue plus profonde de ce que son les cartes de modèle sur le Hub et de comment elles fonctionnent
en arrière plan. Vous pouvez aussi consulter notre [guide de cartes de modèle](../how-to-model-cards)
pour avoir une intuition de la manière dont vous pourriez les utiliser dans vos projets.

## Carte de dépôt

L'objet `RepoCard` est la classe parent de [`ModelCard`], [`DatasetCard`] et `SpaceCard`.

[[autodoc]] huggingface_hub.repocard.RepoCard
    - __init__
    - all

## Donnée de cartes

L'objet [`CardData`] est la classe parent de [`ModelCardData`] et [`DatasetCardData`].

[[autodoc]] huggingface_hub.repocard_data.CardData

## Cartes de modèles

### ModelCard

[[autodoc]] ModelCard

### ModelCardData

[[autodoc]] ModelCardData

## Dataset Cards

Dataset cards are also known as Data Cards in the ML Community.

### DatasetCard

[[autodoc]] DatasetCard

### DatasetCardData

[[autodoc]] DatasetCardData

## Cartes d'espace

### SpaceCard

[[autodoc]] SpaceCard

### SpaceCardData

[[autodoc]] SpaceCardData

## Utilities

### EvalResult

[[autodoc]] EvalResult

### model_index_to_eval_results

[[autodoc]] huggingface_hub.repocard_data.model_index_to_eval_results

### eval_results_to_model_index

[[autodoc]] huggingface_hub.repocard_data.eval_results_to_model_index

### metadata_eval_result

[[autodoc]] huggingface_hub.repocard.metadata_eval_result

### metadata_update

[[autodoc]] huggingface_hub.repocard.metadata_update