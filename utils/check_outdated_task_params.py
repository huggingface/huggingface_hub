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

"""
Utility script to check outdated parameters in task methods and update the InferenceClient task methods arguments
based on the tasks input parameters.

What this script does:
- [x] detect outdated parameters in method signature
- [x] update outdated parameters in method signature
- [ ] detect outdated parameters in method docstrings
- [ ] update outdated parameters in method docstrings
- [ ] detect when parameter not used in method implementation
- [ ] update method implementation when parameter not used
Related resources:
- https://github.com/huggingface/huggingface_hub/issues/2063
- https://github.com/huggingface/huggingface_hub/issues/2557
- https://github.com/huggingface/huggingface_hub/pull/2561
"""

import argparse
import inspect
import re
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Set

import libcst as cst
from helpers import format_source_code

from huggingface_hub.inference._client import InferenceClient


# Paths to project files
BASE_DIR = Path(__file__).parents[1] / "src" / "huggingface_hub"
INFERENCE_TYPES_PATH = BASE_DIR / "inference" / "_generated" / "types"
INFERENCE_CLIENT_FILE = BASE_DIR / "inference" / "_client.py"

DEFAULT_MODULE = "huggingface_hub.inference._generated.types"


# Temporary solution to skip tasks where there is no Parameters dataclass or the schema needs to be updated
TASKS_TO_SKIP = [
    "chat_completion",
    "text_generation",
    "depth_estimation",
    "audio_to_audio",
    "feature_extraction",
    "sentence_similarity",
    "table_question_answering",
    "automatic_speech_recognition",
    "image_to_text",
    "image_to_image",
]

PARAMETERS_DATACLASS_REGEX = re.compile(
    r"""
    ^@dataclass
    \nclass\s(\w+Parameters)\(BaseInferenceType\):
    """,
    re.VERBOSE | re.MULTILINE,
)

CORE_PARAMETERS = {
    "model",  # Model identifier
    "text",  # Text input
    "image",  # Image input
    "audio",  # Audio input
    "inputs",  # Generic inputs
    "input",  # Generic input
    "prompt",  # For generation tasks
    "question",  # For QA tasks
    "context",  # For QA tasks
    "labels",  # For classification tasks
}


#### NODE VISITORS (READING THE CODE)


class DataclassFieldCollector(cst.CSTVisitor):
    """A visitor that collects fields (parameters) from a dataclass."""

    def __init__(self, dataclass_name: str):
        self.dataclass_name = dataclass_name
        self.parameters: Dict[str, Dict[str, str]] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Visit class definitions to find the target dataclass."""

        if node.name.value == self.dataclass_name:
            body_statements = node.body.body
            for index, field in enumerate(body_statements):
                # Check if the statement is a simple statement (like a variable declaration)
                if isinstance(field, cst.SimpleStatementLine):
                    for stmt in field.body:
                        # Check if it's an annotated assignment (typical for dataclass fields)
                        if isinstance(stmt, cst.AnnAssign) and isinstance(stmt.target, cst.Name):
                            param_name = stmt.target.value
                            param_type = cst.Module([]).code_for_node(stmt.annotation.annotation)
                            docstring = self._extract_docstring(body_statements, index)
                            self.parameters[param_name] = {
                                "type": param_type,
                                "docstring": docstring,
                            }

    @staticmethod
    def _extract_docstring(body_statements: List[cst.CSTNode], field_index: int) -> str:
        """Extract the docstring following a field definition."""
        if field_index + 1 < len(body_statements):
            # Check if the next statement is a simple statement (like a string)
            next_stmt = body_statements[field_index + 1]
            if isinstance(next_stmt, cst.SimpleStatementLine):
                for stmt in next_stmt.body:
                    # Check if the statement is a string expression (potential docstring)
                    if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.SimpleString):
                        return stmt.value.evaluated_value.strip()
        # No docstring found or there's no statement after the field
        return ""


class ModulesCollector(cst.CSTVisitor):
    """Visitor that maps type names to their defining modules."""

    def __init__(self):
        self.type_to_module = {}

    def visit_ClassDef(self, node: cst.ClassDef):
        """Map class definitions to the current module."""
        self.type_to_module[node.name.value] = DEFAULT_MODULE

    def visit_ImportFrom(self, node: cst.ImportFrom):
        """Map imported types to their modules."""
        if node.module:
            module_name = node.module.value
            for alias in node.names:
                self.type_to_module[alias.name.value] = module_name


class ArgumentsCollector(cst.CSTVisitor):
    """Collects existing argument names from a method."""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.existing_args: Set[str] = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if node.name.value == self.method_name:
            self.existing_args.update(
                param.name.value
                for param in node.params.params + node.params.kwonly_params
                if param.name.value != "self"
            )


class DeprecatedParamsCollector(cst.CSTVisitor):
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.deprecated_params: Set[str] = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if node.name.value == self.method_name:
            for decorator in node.decorators:
                if (
                    isinstance(decorator.decorator, cst.Call)
                    and decorator.decorator.func.value == "_deprecate_arguments"
                ):
                    for arg in decorator.decorator.args:
                        if arg.keyword and arg.keyword.value == "deprecated_args":
                            if isinstance(arg.value, cst.List):
                                for element in arg.value.elements:
                                    if isinstance(element.value, cst.SimpleString):
                                        self.deprecated_params.add(element.value.evaluated_value.strip('"'))


#### TREE TRANSFORMERS (UPDATING THE CODE)


class HandleOutdatedParameters(cst.CSTTransformer):
    """Handles outdated parameters by either removing them or marking them as deprecated using a decorator."""

    def __init__(
        self,
        method_name: str,
        outdated_params: Set[str],
        action: str,
        version: Optional[str],
    ):
        self.method_name = method_name
        self.in_target_function = False
        self.outdated_params = outdated_params
        self.action = action
        self.version = version
        assert self.action in ("remove", "deprecate"), "Invalid action, must be 'remove' or 'deprecate'"
        if self.action == "deprecate":
            assert self.version is not None, "Version is required when deprecating parameters"

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if original_node.name.value != self.method_name:
            return updated_node
        self.in_target_function = True
        if self.action == "remove":
            # Remove outdated parameters completely
            raise NotImplementedError("Removing parameters is not yet implemented")

        else:  # DEPRECATE
            # Add deprecation decorator
            decorator = cst.Decorator(
                decorator=cst.Call(
                    func=cst.Name("_deprecate_arguments"),
                    args=[
                        cst.Arg(
                            value=cst.SimpleString(f'"{self.version}"'),  # Use the configurable version
                            keyword=cst.Name("version"),
                        ),
                        cst.Arg(
                            value=cst.List(
                                elements=[
                                    cst.Element(value=cst.SimpleString(f'"{param}"')) for param in self.outdated_params
                                ]
                            ),
                            keyword=cst.Name("deprecated_args"),
                        ),
                        cst.Arg(
                            value=cst.SimpleString(
                                f'"The {", ".join(f"`{p}`" for p in self.outdated_params)} '
                                f'{"parameter is" if len(self.outdated_params) == 1 else "parameters are"} '
                                f'deprecated and will be removed in version {self.version}."'
                            ),
                            keyword=cst.Name("custom_message"),
                        ),
                    ],
                )
            )
            # Add decorator to existing decorators
            decorators = list(updated_node.decorators) if updated_node.decorators else []
            decorators.append(decorator)

            return updated_node.with_changes(decorators=decorators)


#### UTILS


def check_outdated_parameters(
    inference_client_module: cst.Module,
    parameters_module: cst.Module,
    method_name: str,
    parameter_type_name: str,
) -> Set[str]:
    """
    Check for parameters that exist in the method signature but not in the parameters dataclass.
    Excludes core parameters that are always valid.
    """
    # Get valid parameters from the parameters dataclass
    params_collector = DataclassFieldCollector(parameter_type_name)
    parameters_module.visit(params_collector)
    valid_params = set(params_collector.parameters.keys())

    # Get existing arguments using the existing ArgumentsCollector
    method_argument_collector = ArgumentsCollector(method_name)
    inference_client_module.visit(method_argument_collector)

    # Find outdated parameters, excluding core parameters and already deprecated parameters
    method_params = method_argument_collector.existing_args
    deprecated_params_collector = DeprecatedParamsCollector(method_name)
    inference_client_module.visit(deprecated_params_collector)
    deprecated_params = deprecated_params_collector.deprecated_params
    return method_params - valid_params - CORE_PARAMETERS - deprecated_params


def update_outdated_parameters(
    module: cst.Module,
    method_name: str,
    outdated_params: Set[str],
    action: str,
    version: Optional[str],
) -> cst.Module:
    """
    Update the method signature for outdated parameters.

    Args:
        module: The module to update
        method_name: Name of the method to update
        outdated_params: Set of parameter names to handle
        action: Whether to remove or deprecate the parameters
    """
    transformer = HandleOutdatedParameters(method_name, outdated_params, action, version=version)
    return module.visit(transformer)


def _get_parameter_type_name(method_name: str) -> Optional[str]:
    file_path = INFERENCE_TYPES_PATH / f"{method_name}.py"
    if not file_path.is_file():
        print(f"File not found: {file_path}")
        return None

    content = file_path.read_text(encoding="utf-8")
    match = PARAMETERS_DATACLASS_REGEX.search(content)

    return match.group(1) if match else None


def _parse_module_from_file(filepath: Path) -> Optional[cst.Module]:
    try:
        code = filepath.read_text(encoding="utf-8")
        return cst.parse_module(code)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except cst.ParserSyntaxError as e:
        print(f"Syntax error while parsing {filepath}: {e}")
    return None


def _check_parameters(
    method_params: Dict[str, str],
    update: bool,
    action: str,
    version: Optional[str],
) -> NoReturn:
    """
    Check if task methods have outdated parameters and update the InferenceClient source code if needed.
    """
    logs = []
    inference_client_filename = INFERENCE_CLIENT_FILE
    # Read and parse the inference client module
    inference_client_module = _parse_module_from_file(inference_client_filename)
    modified_module = inference_client_module
    has_changes = False

    for method_name, parameter_type_name in method_params.items():
        parameters_filename = INFERENCE_TYPES_PATH / f"{method_name}.py"
        parameters_module = _parse_module_from_file(parameters_filename)

        # Check for outdated parameters
        outdated_params = check_outdated_parameters(
            modified_module, parameters_module, method_name, parameter_type_name
        )

        if not outdated_params:
            continue

        if update:
            if outdated_params:
                # Remove outdated parameters
                modified_module = update_outdated_parameters(
                    modified_module, method_name, outdated_params, action=action, version=version
                )

            has_changes = True
        else:
            if outdated_params:
                logs.append(f"❌ Outdated parameters found in `{method_name}`: {', '.join(outdated_params)}")

    if has_changes:
        # Format the updated source code
        formatted_source_code = format_source_code(modified_module.code)
        INFERENCE_CLIENT_FILE.write_text(formatted_source_code)

    if len(logs) > 0:
        for log in logs:
            print(log)
        print(
            "❌ Mismatch between between parameters defined in tasks methods signature in "
            "`./src/huggingface_hub/inference/_client.py` and parameters defined in "
            "`./src/huggingface_hub/inference/_generated/types.py \n"
            "Please run `make inference_update` or `python utils/generate_task_parameters.py --update"
        )
        exit(1)
    else:
        if update:
            print(
                "✅ InferenceClient source code has been updated in"
                " `./src/huggingface_hub/inference/_client.py`.\n   Please make sure the changes are"
                " accurate and commit them."
            )
        else:
            print("✅ All good!")
        exit(0)


def update_inference_client(update: bool, action: str, version: Optional[str]):
    print(f"🙈 Skipping the following tasks: {TASKS_TO_SKIP}")
    # Get all tasks from the ./src/huggingface_hub/inference/_generated/types/
    tasks = set()
    for file in INFERENCE_TYPES_PATH.glob("*.py"):
        if file.stem not in TASKS_TO_SKIP:
            tasks.add(file.stem)

    # Construct a mapping between method names and their parameters dataclass names
    method_params = {}
    for method_name, _ in inspect.getmembers(InferenceClient, predicate=inspect.isfunction):
        if method_name.startswith("_") or method_name not in tasks:
            continue
        parameter_type_name = _get_parameter_type_name(method_name)
        if parameter_type_name is not None:
            method_params[method_name] = parameter_type_name
    _check_parameters(method_params, update=update, action=action, version=version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help=("Whether to update `./src/huggingface_hub/inference/_client.py` if parameters are outdated."),
    )
    parser.add_argument(
        "--action",
        type=str,
        help=("Whether to remove or deprecate the parameters."),
    )
    parser.add_argument(
        "--version",
        type=str,
        help=("The version of the library to use for deprecating parameters."),
    )
    args = parser.parse_args()
    update_inference_client(update=args.update, action=args.action, version=args.version)
