# coding=utf-8
# Copyright 2024-present, the HuggingFace Inc. team.
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
"""Utility script to check consistency between input parameters of InferenceClient methods and generated types.

TODO: check all methods
TODO: check parameters types
TODO: check parameters default values
TODO: check parameters (type, description) are consistent in the docstrings
TODO: (low priority) automatically generate the input types from the methods
"""

import inspect
from dataclasses import is_dataclass
from typing import Any, get_args

from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated import types


METHODS_TO_SKIP = [
    # + all private methods
    "post",
    "conversational",
]
PARAMETERS_TO_SKIP = {
    "chat_completion": {},
    "text_generation": {
        "stop",  # stop_sequence instead (for legacy reasons)
    },
}
UNDOCUMENTED_PARAMETERS = {
    "self",
}


def check_method(method_name: str, method: Any):
    input_type_name = "".join(part.capitalize() for part in method_name.split("_")) + "Input"
    if not hasattr(types, input_type_name):
        return [f"Missing input type for method {method_name}"]

    input_type = getattr(types, input_type_name)
    docstring = method.__doc__

    if method_name == "chat_completion":
        # Special case for chat_completion
        parameters_type = input_type
    else:
        parameters_field = input_type.__dataclass_fields__.get("parameters", None)
        if parameters_field is None:
            return [f"Missing 'parameters' field for type {input_type}"]

        parameters_type = get_args(parameters_field.type)[0]
        if not is_dataclass(parameters_type):
            return [f"'parameters' field is not a dataclass for type {input_type} ({parameters_type})"]

    # For each expected parameter, check it is defined
    logs = []
    method_params = inspect.signature(method).parameters
    for param_name in parameters_type.__dataclass_fields__:
        if param_name in PARAMETERS_TO_SKIP[method_name]:
            continue
        if param_name not in method_params:
            logs.append(f"Missing parameter {param_name} in method signature")

    # Check parameter is documented in docstring
    for param_name in method_params:
        if param_name in UNDOCUMENTED_PARAMETERS:
            continue
        if param_name in PARAMETERS_TO_SKIP[method_name]:
            continue
        if f"{param_name} (" not in docstring:
            logs.append(f"Parameter {param_name} is not documented")

    return logs


# Inspect InferenceClient methods individually
exit_code = 0
all_logs = []  # print details only if errors are found
for method_name, method in inspect.getmembers(InferenceClient, predicate=inspect.isfunction):
    if method_name.startswith("_") or method_name in METHODS_TO_SKIP:
        continue
    if method_name not in PARAMETERS_TO_SKIP:
        all_logs.append(f"  ⏩️ {method_name}: skipped")
        continue

    logs = check_method(method_name, method)
    if len(logs) > 0:
        exit_code = 1
        all_logs.append(f"  ❌ {method_name}: errors found")
        all_logs.append("\n".join(" " * 4 + log for log in logs))
        continue
    else:
        all_logs.append(f"  ✅ {method_name}: success!")
        continue

if exit_code == 0:
    print("✅ All good! (inference inputs)")
else:
    print("❌ Inconsistency found in inference inputs.")
    for log in all_logs:
        print(log)
exit(exit_code)
