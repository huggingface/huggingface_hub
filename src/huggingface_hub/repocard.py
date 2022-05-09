import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repocard_types import (
    ModelIndex,
    SingleMetric,
    SingleResult,
    SingleResultDataset,
    SingleResultTask,
)

from .constants import REPOCARD_NAME


# exact same regex as in the Hub server. Please keep in sync.
REGEX_YAML_BLOCK = re.compile(r"---[\n\r]+([\S\s]*?)[\n\r]+---[\n\r]")

UNIQUE_RESULT_FEATURES = ["dataset", "task"]
UNIQUE_METRIC_FEATURES = ["name", "type"]


def metadata_load(local_path: Union[str, Path]) -> Optional[Dict]:
    content = Path(local_path).read_text()
    match = REGEX_YAML_BLOCK.search(content)
    if match:
        yaml_block = match.group(1)
        data = yaml.safe_load(yaml_block)
        if isinstance(data, dict):
            return data
        else:
            raise ValueError("repo card metadata block should be a dict")
    else:
        return None


def metadata_save(local_path: Union[str, Path], data: Dict) -> None:
    """
    Save the metadata dict in the upper YAML part Trying to preserve newlines as
    in the existing file. Docs about open() with newline="" parameter:
    https://docs.python.org/3/library/functions.html?highlight=open#open Does
    not work with "^M" linebreaks, which are replaced by \n
    """
    line_break = "\n"
    content = ""
    # try to detect existing newline character
    if os.path.exists(local_path):
        with open(local_path, "r", newline="") as readme:
            if type(readme.newlines) is tuple:
                line_break = readme.newlines[0]
            if type(readme.newlines) is str:
                line_break = readme.newlines
            content = readme.read()

    # creates a new file if it not
    with open(local_path, "w", newline="") as readme:
        data_yaml = yaml.dump(data, sort_keys=False, line_break=line_break)
        # sort_keys: keep dict order
        match = REGEX_YAML_BLOCK.search(content)
        if match:
            output = (
                content[: match.start()]
                + f"---{line_break}{data_yaml}---{line_break}"
                + content[match.end() :]
            )
        else:
            output = f"---{line_break}{data_yaml}---{line_break}{content}"

        readme.write(output)
        readme.close()


def metadata_eval_result(
    model_pretty_name: str,
    task_pretty_name: str,
    task_id: str,
    metrics_pretty_name: str,
    metrics_id: str,
    metrics_value: Any,
    dataset_pretty_name: str,
    dataset_id: str,
) -> Dict:
    model_index = ModelIndex(
        name=model_pretty_name,
        results=[
            SingleResult(
                metrics=[
                    SingleMetric(
                        type=metrics_id,
                        name=metrics_pretty_name,
                        value=metrics_value,
                    ),
                ],
                task=SingleResultTask(type=task_id, name=task_pretty_name),
                dataset=SingleResultDataset(name=dataset_pretty_name, type=dataset_id),
            )
        ],
    )
    # use `dict_factory` to recursively ignore None values
    data = dataclasses.asdict(
        model_index, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}
    )
    return {"model-index": [data]}


def metadata_update(
    repo_id: str,
    metadata: Dict,
    *,
    repo_type: str = None,
    overwrite: bool = False,
    token: str = None,
) -> str:
    """
    Updates the metadata in the README.md of a repository on the Hugging Face Hub.

    Example:
    >>> from huggingface_hub import metadata_update
    >>> metadata = {'model-index': [{'name': 'RoBERTa fine-tuned on ReactionGIF',
    ...             'results': [{'dataset': {'name': 'ReactionGIF',
    ...                                      'type': 'julien-c/reactiongif'},
    ...                           'metrics': [{'name': 'Recall',
    ...                                        'type': 'recall',
    ...                                        'value': 0.7762102282047272}],
    ...                          'task': {'name': 'Text Classification',
    ...                                   'type': 'text-classification'}}]}]}
    >>> update_metdata("julien-c/reactiongif-roberta", metadata)

    Args:
        repo_id (`str`):
            The name of the repository.
        metadata (`dict`):
            A dictionary containing the metadata to be updated.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if updating to a dataset or space,
            `None` or `"model"` if updating to a model. Default is `None`.
        overwrite (`bool`, *optional*, defaults to `False`):
            If set to `True` an existing field can be overwritten, otherwise
            attempting to overwrite an existing field will cause an error.
        token (`str`, *optional*):
            The Hugging Face authentication token.

    Returns:
        `str`: URL of the commit which updated the card metadata.
    """

    filepath = hf_hub_download(
        repo_id,
        filename=REPOCARD_NAME,
        repo_type=repo_type,
        use_auth_token=token,
        force_download=True,
    )
    existing_metadata = metadata_load(filepath)

    for key in metadata:
        # update model index containing the evaluation results
        if key == "model-index":
            if "model-index" not in existing_metadata:
                existing_metadata["model-index"] = metadata["model-index"]
            else:
                # the model-index contains a list of results as used by PwC but only has one element thus we take the first one
                existing_metadata["model-index"][0][
                    "results"
                ] = _update_metadata_model_index(
                    existing_metadata["model-index"][0]["results"],
                    metadata["model-index"][0]["results"],
                    overwrite=overwrite,
                )
        # update all fields except model index
        else:
            if key in existing_metadata and not overwrite:
                if existing_metadata[key] != metadata[key]:
                    raise ValueError(
                        f"""You passed a new value for the existing meta data field '{key}'. Set `overwrite=True` to overwrite existing metadata."""
                    )
            else:
                existing_metadata[key] = metadata[key]

    # save and push to hub
    metadata_save(filepath, existing_metadata)

    return HfApi().upload_file(
        path_or_fileobj=filepath,
        path_in_repo=REPOCARD_NAME,
        repo_id=repo_id,
        repo_type=repo_type,
        identical_ok=False,
        token=token,
    )


def _update_metadata_model_index(existing_results, new_results, overwrite=False):
    """
    Updates the model-index fields in the metadata. If results with same unique
    features exist they are updated, else a new result is appended. Updating existing
    values is only possible if `overwrite=True`.

    Args:
        new_metrics (`List[dict]`):
            List of new metadata results.
        existing_metrics (`List[dict]`):
            List of existing metadata results.
        overwrite (`bool`, *optional*, defaults to `False`):
            If set to `True`, an existing metric values can be overwritten, otherwise
            attempting to overwrite an existing field will cause an error.

    Returns:
        `list`: List of updated metadata results
    """
    for new_result in new_results:
        result_found = False
        for existing_result_index, existing_result in enumerate(existing_results):
            if all(
                new_result[feat] == existing_result[feat]
                for feat in UNIQUE_RESULT_FEATURES
            ):
                result_found = True
                existing_results[existing_result_index][
                    "metrics"
                ] = _update_metadata_results_metric(
                    new_result["metrics"],
                    existing_result["metrics"],
                    overwrite=overwrite,
                )
        if not result_found:
            existing_results.append(new_result)
    return existing_results


def _update_metadata_results_metric(new_metrics, existing_metrics, overwrite=False):
    """
    Updates the metrics list of a result in the metadata. If metrics with same unique
    features exist their values are updated, else a new metric is appended. Updating
    existing values is only possible if `overwrite=True`.

    Args:
        new_metrics (`list`):
            List of new metrics.
        existing_metrics (`list`):
            List of existing metrics.
        overwrite (`bool`, *optional*, defaults to `False`):
            If set to `True`, an existing metric values can be overwritten, otherwise
            attempting to overwrite an existing field will cause an error.

    Returns:
        `list`: List of updated metrics
    """
    for new_metric in new_metrics:
        metric_exists = False
        for existing_metric_index, existing_metric in enumerate(existing_metrics):
            if all(
                new_metric[feat] == existing_metric[feat]
                for feat in UNIQUE_METRIC_FEATURES
            ):
                if overwrite:
                    existing_metrics[existing_metric_index]["value"] = new_metric[
                        "value"
                    ]
                else:
                    # if metric exists and value is not the same throw an error without overwrite flag
                    if (
                        existing_metrics[existing_metric_index]["value"]
                        != new_metric["value"]
                    ):
                        existing_str = ", ".join(
                            f"{feat}: {new_metric[feat]}"
                            for feat in UNIQUE_METRIC_FEATURES
                        )
                        raise ValueError(
                            "You passed a new value for the existing metric"
                            f" '{existing_str}'. Set `overwrite=True` to overwrite"
                            " existing metrics."
                        )
                metric_exists = True
        if not metric_exists:
            existing_metrics.append(new_metric)
    return existing_metrics
