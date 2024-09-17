from configparser import ConfigParser
from pathlib import Path

from .. import constants


def _read_profiles() -> ConfigParser:
    """
    Returns the parsed INI file containing the auth profiles.
    The file is located at `HF_PROFILES_PATH`, defaulting to `~/.cache/huggingface/profiles`.
    If the file does not exist, it will be created.

    Returns:
        `ConfigParser`: The configuration parser object containing the parsed INI file.
    """
    config = ConfigParser()
    profiles_path = Path(constants.HF_PROFILES_PATH)
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.touch(exist_ok=True)
    config.read([profiles_path])
    return config


def _save_profiles(config: ConfigParser) -> None:
    """
    Saves the given configuration to the profiles file.

    Args:
        config (`ConfigParser`):
            The configuration parser object to save.
    """
    profiles_path = Path(constants.HF_PROFILES_PATH)
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    with profiles_path.open("w+") as config_file:
        config.write(config_file)
