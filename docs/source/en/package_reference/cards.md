<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Repository Cards

The huggingface_hub library provides a Python interface to create, share, and update Model/Dataset Cards.
Visit the [dedicated documentation page](https://huggingface.co/docs/hub/models-cards) for a deeper view of what
Model Cards on the Hub are, and how they work under the hood. You can also check out our [Model Cards guide](../how-to-model-cards) to
get a feel for how you would use these utilities in your own projects.

## Repo Card

The `RepoCard` object is the parent class of [`ModelCard`], [`DatasetCard`] and `SpaceCard`.

[[autodoc]] huggingface_hub.repocard.RepoCard
    - __init__
    - all

## Card Data

The [`CardData`] object is the parent class of [`ModelCardData`] and [`DatasetCardData`].

[[autodoc]] huggingface_hub.repocard_data.CardData

## Model Cards

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

## Space Cards

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
