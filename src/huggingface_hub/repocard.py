import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
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
    Save the metadata dict in the upper YAML part
    Trying to preserve newlines as in the existing file.
    Docs about open() with newline="" parameter:
    https://docs.python.org/3/library/functions.html?highlight=open#open
    Does not work with "^M" linebreaks, which are replaced by \n
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
