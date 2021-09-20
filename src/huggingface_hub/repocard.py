import re
from pathlib import Path
from typing import Dict, Optional, Union

import yaml


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
    # try to preserve newlines
    linebrk = "\n"
    # this is known not to work with ^M linebreaks, so ^M are replaced by \n
    with open(local_path, "r", newline="") as readme:
        if type(readme.newlines) is tuple:
            linebrk = readme.newlines[0]
        if type(readme.newlines) is str:
            linebrk = readme.newlines
        content = readme.read()

    if content:
        with open(local_path, "w", newline="") as readme:
            data_yaml = yaml.dump(data, sort_keys=False, line_break=linebrk)
            # sort_keys: keep dict order
            match = REGEX_YAML_BLOCK.search(content)
            if match:
                output = (
                    content[: match.start()]
                    + f"---{linebrk}{data_yaml}---{linebrk}"
                    + content[match.end() :]
                )
            else:
                output = f"---{linebrk}{data_yaml}---{linebrk}{content}"

            readme.write(output)
            readme.close()
