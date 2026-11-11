# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modality taxonomy used to filter repositories by modality instead of by sub-task.

A "modality" is a broad family of tasks (text, image, audio, video, ...) while a "task" is a
specific sub-task (`text-classification`, `depth-estimation`, ...). Filtering by modality lets you
browse everything related to, say, audio, without listing each audio task one by one.

The Hub exposes modalities differently depending on the repo type:

- **datasets** declare their modalities as `modality:*` tags (`modality:audio`, `modality:text`, ...),
  listed by [`HfApi.get_dataset_tags`]. The Hub API can filter on them directly.
- **models** have no modality field: a model only declares a `pipeline_tag` (its task). The Hub groups
  those pipeline tags by `subType` in [`HfApi.get_model_tags`] (`nlp`, `cv`, `audio`, `multimodal`,
  `rl`, `tabular`, `other`). To filter models by modality, we map the modality to its pipeline tags and
  query each of them (see [`HfApi.list_models`]).

The mapping below mirrors that Hub grouping. Broad modalities map to a Hub `subType` as-is, and a few
narrower modalities (`video`, `3d`, `document`, `timeseries`) are subsets of one of them. `tests/test_hf_api.py`
checks the mapping against the Hub so that it stays in sync when a task is added.
"""

from typing import Literal, get_args


# Canonical modalities accepted by `HfApi.list_models` and `HfApi.list_datasets`.
Modality_T = Literal[
    "text",
    "image",
    "audio",
    "video",
    "3d",
    "multimodal",
    "document",
    "tabular",
    "timeseries",
    "geospatial",
    "reinforcement-learning",
    "other",
]

MODALITIES: tuple[str, ...] = get_args(Modality_T)

# Convenient spellings mapped to their canonical modality.
MODALITY_ALIASES: dict[str, str] = {
    "nlp": "text",
    "natural-language-processing": "text",
    "natural_language_processing": "text",
    "vision": "image",
    "cv": "image",
    "computer-vision": "image",
    "computer_vision": "image",
    "speech": "audio",
    "multi-modal": "multimodal",
    "multi_modal": "multimodal",
    "time-series": "timeseries",
    "time_series": "timeseries",
    "rl": "reinforcement-learning",
    "reinforcement_learning": "reinforcement-learning",
}

# Modality -> pipeline tags (i.e. tasks) belonging to it. `text`, `image`, `audio`, `multimodal`,
# `tabular`, `reinforcement-learning` and `other` are the Hub `subType`s and together cover every
# pipeline tag of the Hub; `video`, `3d`, `document` and `timeseries` are subsets of them.
MODEL_MODALITY_TO_PIPELINE_TAGS: dict[str, tuple[str, ...]] = {
    # Hub subType "nlp"
    "text": (
        "feature-extraction",
        "fill-mask",
        "question-answering",
        "sentence-similarity",
        "summarization",
        "table-question-answering",
        "text-classification",
        "text-generation",
        "text-ranking",
        "token-classification",
        "translation",
        "zero-shot-classification",
    ),
    # Hub subType "cv"
    "image": (
        "depth-estimation",
        "image-classification",
        "image-feature-extraction",
        "image-segmentation",
        "image-to-image",
        "image-to-text",
        "image-to-video",
        "keypoint-detection",
        "mask-generation",
        "object-detection",
        "text-to-image",
        "text-to-3d",
        "text-to-video",
        "unconditional-image-generation",
        "video-classification",
        "video-to-video",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
        "image-to-3d",
    ),
    # Hub subType "audio"
    "audio": (
        "audio-classification",
        "audio-to-audio",
        "automatic-speech-recognition",
        "text-to-audio",
        "text-to-speech",
        "voice-activity-detection",
    ),
    # Hub subType "multimodal"
    "multimodal": (
        "any-to-any",
        "audio-text-to-text",
        "document-question-answering",
        "image-text-to-image",
        "image-text-to-text",
        "image-text-to-video",
        "video-text-to-text",
        "visual-document-retrieval",
        "visual-question-answering",
    ),
    # Hub subType "tabular"
    "tabular": (
        "tabular-classification",
        "tabular-regression",
        "time-series-forecasting",
    ),
    # Hub subType "rl"
    "reinforcement-learning": (
        "reinforcement-learning",
        "robotics",
    ),
    # Hub subType "other"
    "other": ("graph-ml",),
    # Subsets
    "video": (
        "image-to-video",
        "text-to-video",
        "video-classification",
        "video-to-video",
    ),
    "3d": (
        "image-to-3d",
        "text-to-3d",
    ),
    "document": (
        "document-question-answering",
        "visual-document-retrieval",
    ),
    "timeseries": ("time-series-forecasting",),
}

# Modality -> `modality:*` tag used by dataset cards. `multimodal`, `reinforcement-learning` and `other`
# have no equivalent: a multimodal dataset simply declares several modality tags, and the Hub API does
# not support OR-ing tags.
DATASET_MODALITY_TAGS: dict[str, str] = {
    "text": "modality:text",
    "image": "modality:image",
    "audio": "modality:audio",
    "video": "modality:video",
    "3d": "modality:3d",
    "tabular": "modality:tabular",
    "document": "modality:document",
    "timeseries": "modality:timeseries",
    "geospatial": "modality:geospatial",
}


def resolve_modality(modality: str) -> str:
    """Normalize a modality (case-insensitive, aliases resolved) and validate it.

    Raises:
        ValueError: If `modality` is not a known modality.
    """
    if not isinstance(modality, str):
        raise ValueError(f"`modality` must be a string, got {type(modality)}.")
    key = modality.strip().lower().replace(" ", "-")
    key = MODALITY_ALIASES.get(key, key)
    if key not in MODALITIES:
        raise ValueError(
            f"Invalid modality {modality!r}. Expected one of: {', '.join(MODALITIES)}. "
            "Common aliases such as 'nlp', 'computer-vision' or 'speech' are also accepted."
        )
    return key


def modality_to_pipeline_tags(modality: str) -> tuple[str, ...]:
    """Return the pipeline tags to query on the Hub to list models of a given modality."""
    key = resolve_modality(modality)
    pipeline_tags = MODEL_MODALITY_TO_PIPELINE_TAGS.get(key)
    if pipeline_tags is None:
        supported = ", ".join(sorted(MODEL_MODALITY_TO_PIPELINE_TAGS))
        raise ValueError(
            f"Modality {key!r} cannot be used to filter models: no pipeline tag belongs to it. "
            f"Supported modalities for models are: {supported}."
        )
    return pipeline_tags


def modality_to_dataset_tag(modality: str) -> str:
    """Return the `modality:*` filter to pass to the Hub API to list datasets of a given modality."""
    key = resolve_modality(modality)
    tag = DATASET_MODALITY_TAGS.get(key)
    if tag is None:
        supported = ", ".join(sorted(DATASET_MODALITY_TAGS))
        raise ValueError(
            f"Modality {key!r} cannot be used to filter datasets. Supported modalities for datasets are: {supported}. "
            "For datasets that mix several modalities, filter on one modality or pass the raw tags with `filter=`."
        )
    return tag
