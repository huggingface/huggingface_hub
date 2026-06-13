<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Repository Cards

La bibliothèque huggingface_hub fournit une interface Python pour créer, partager et mettre à jour les Model/Dataset Cards.
Consultez la [page de documentation dédiée](https://huggingface.co/docs/hub/models-cards) pour une vue plus approfondie de ce que
sont les Model Cards sur le Hub et comment elles fonctionnent sous le capot. Vous pouvez également consulter notre [guide Model Cards](../how-to-model-cards) pour
avoir une idée de comment vous utiliseriez ces utilitaires dans vos propres projets.

## Repo Card

L'objet `RepoCard` est la classe parente de [`ModelCard`], [`DatasetCard`] et `SpaceCard`.

[[autodoc]] huggingface_hub.repocard.RepoCard
    - __init__
    - all

## Card Data

L'objet [`CardData`] est la classe parente de [`ModelCardData`] et [`DatasetCardData`].

[[autodoc]] huggingface_hub.repocard_data.CardData

## Model Cards

### ModelCard

[[autodoc]] ModelCard

### ModelCardData

[[autodoc]] ModelCardData

## Dataset Cards

Les Dataset cards sont également connues sous le nom de Data Cards dans la communauté ML.

### DatasetCard

[[autodoc]] DatasetCard

### DatasetCardData

[[autodoc]] DatasetCardData

## Space Cards

### SpaceCard

[[autodoc]] SpaceCard

### SpaceCardData

[[autodoc]] SpaceCardData

## Utilitaires

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
