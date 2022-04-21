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


# exact same regex as in the Hub server. Please keep in sync.
REGEX_YAML_BLOCK = re.compile(r"---[\n\r]+([\S\s]*?)[\n\r]+---[\n\r]")


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
    repo_type: str = None,
    overwrite: bool = False,
    token: str = None,
) -> None:
    """
    Updates the metadata in the README.md of a repository on the Hugging Face Hub.

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
            The Hugging Face authentication token
    """

    filepath = hf_hub_download(
        repo_id,
        filename="README.md",
        repo_type=repo_type,
        use_auth_token=token,
        force_download=True,
    )
    existing_metadata = metadata_load(filepath)

    for key in metadata:
        # update all fields except model index
        if key != "model-index":
            if key in existing_metadata and not overwrite:
                if existing_metadata[key] != metadata[key]:
                    raise ValueError(
                        f"""You passed a new value for the existing meta data field '{key}'. Set `overwrite=True` to overwrite existing metadata."""
                    )
            else:
                existing_metadata[key] = metadata[key]
        # update model index containing the evaluation results
        else:
            if "model-index" not in existing_metadata:
                existing_metadata["model-index"] = metadata["model-index"]
            else:
                existing_metadata["model-index"][0][
                    "results"
                ] = _update_metadata_model_index(
                    existing_metadata["model-index"][0]["results"],
                    metadata["model-index"][0]["results"],
                    overwrite=overwrite,
                )

    # save and push to hub
    metadata_save(filepath, existing_metadata)

    HfApi().upload_file(
        path_or_fileobj=filepath,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        identical_ok=True,
        token=token,
    )


def _update_metadata_model_index(existing_results, new_results, overwrite=False):
    for new_result in new_results:
        result_found = False
        for existing_result_index, existing_result in enumerate(existing_results):
            if (
                new_result["dataset"] == existing_result["dataset"]
                and new_result["task"] == existing_result["task"]
            ):
                result_found = True
                for new_metric in new_result["metrics"]:
                    metric_exists = False
                    for existing_metric_index, existing_metric in enumerate(
                        existing_result["metrics"]
                    ):
                        if (
                            new_metric["name"] == existing_metric["name"]
                            and new_metric["type"] == existing_metric["type"]
                        ):
                            if overwrite:
                                existing_results[existing_result_index]["metrics"][
                                    existing_metric_index
                                ]["value"] = new_metric["value"]
                            else:
                                # if metric exists and value is not the same throw an error without overwrite flag
                                if (
                                    existing_results[existing_result_index]["metrics"][
                                        existing_metric_index
                                    ]["value"]
                                    != new_metric["value"]
                                ):
                                    raise ValueError(
                                        f"""You passed a new value for the existing metric '{new_metric["name"]}'. Set `overwrite=True` to overwrite existing metrics."""
                                    )
                            metric_exists = True
                    if not metric_exists:
                        existing_results[existing_result_index]["metrics"].append(
                            new_metric
                        )
        if not result_found:
            existing_results.append(new_result)
    return existing_results
