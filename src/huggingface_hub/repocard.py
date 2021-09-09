import re
from pathlib import Path
from typing import Dict, Optional, Union

import yaml


REGEX_YAML_BLOCK = re.compile(r"---[\n\r]+([\S\s]*?)[\n\r]+---[\n\r]")
# exact same regex as in the hf hub server. Please keep in sync.


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
    data_yaml = yaml.dump(data, sort_keys=False)
    # keep dict order
    content = Path(local_path).read_text()
    match = REGEX_YAML_BLOCK.search(content)
    if match:
        output = (
            content[: match.start()] + f"---\n{data_yaml}---\n" + content[match.end() :]
        )
    else:
        output = f"---\n{data_yaml}---\n{content}"

    Path(local_path).write_text(output)
