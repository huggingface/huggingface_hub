# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
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
""" Logging utilities. """

import logging
import os
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _get_default_logging_level():
    """
    If HUGGINGFACE_HUB_VERBOSITY env var is set to one of the valid choices return that as the new default level.
    If it is not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv("HUGGINGFACE_HUB_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option HUGGINGFACE_HUB_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _configure_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.addHandler(logging.StreamHandler())
    library_root_logger.setLevel(_get_default_logging_level())


def _reset_library_root_logger() -> None:
    library_root_logger = _get_library_root_logger()
    library_root_logger.setLevel(logging.NOTSET)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.
    This function is not supposed to be directly accessed by library users.
    """

    if name is None:
        name = _get_library_name()

    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the HuggingFace Hub's root logger.
    Returns:
        Logging level, e.g., ``huggingface_hub.logging.DEBUG`` and ``huggingface_hub.logging.INFO``.
    .. note::
        HuggingFace Hub has following logging levels:
        - ``huggingface_hub.logging.CRITICAL``, ``huggingface_hub.logging.FATAL``
        - ``huggingface_hub.logging.ERROR``
        - ``huggingface_hub.logging.WARNING``, ``huggingface_hub.logging.WARN``
        - ``huggingface_hub.logging.INFO``
        - ``huggingface_hub.logging.DEBUG``
    """
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """Set the level for the HuggingFace Hub's root logger.
    Args:
        verbosity:
            Logging level, e.g., ``huggingface_hub.logging.DEBUG`` and ``huggingface_hub.logging.INFO``.
    """
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    return set_verbosity(INFO)


def set_verbosity_warning():
    return set_verbosity(WARNING)


def set_verbosity_debug():
    return set_verbosity(DEBUG)


def set_verbosity_error():
    return set_verbosity(ERROR)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.
    Note that log propagation is disabled by default.
    """
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.
    Please disable the HuggingFace Hub's default handler to prevent double logging if the root logger has
    been configured.
    """
    _get_library_root_logger().propagate = True


_configure_library_root_logger()
