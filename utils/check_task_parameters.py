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
Utility script to check and update the InferenceClient task methods arguments and docstrings
based on the tasks input parameters.

What this script does:
- [x] detect missing parameters in method signature
- [x] add missing parameters to methods signature
- [x] detect missing parameters in method docstrings
- [x] add missing parameters to methods docstrings
- [x] detect outdated parameters in method signature
- [x] update outdated parameters in method signature
- [x] detect outdated parameters in method docstrings
- [x] update outdated parameters in method docstrings
- [ ] detect when parameter not used in method implementation
- [ ] update method implementation when parameter not used
Related resources:
- https://github.com/huggingface/huggingface_hub/issues/2063
- https://github.com/huggingface/huggingface_hub/issues/2557
- https://github.com/huggingface/huggingface_hub/pull/2561
"""

import argparse
import builtins
import inspect
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, NoReturn, Optional

import libcst as cst
from helpers import format_source_code
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import GatherImportsVisitor

from huggingface_hub import InferenceClient


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
    "automatic_speech_recognition",
    "image_to_text",
]

PARAMETERS_DATACLASS_REGEX = re.compile(
    r"""
    ^@dataclass_with_extra
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
    "extra_body",  # For extra parameters
}

#### NODE VISITORS (READING THE CODE)


class DataclassFieldCollector(cst.CSTVisitor):
    """A visitor that collects fields (parameters) from a dataclass."""

    def __init__(self, dataclass_name: str):
        self.dataclass_name = dataclass_name
        self.parameters: dict[str, dict[str, str]] = {}

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
                            # Check if there's a default value
                            has_default = stmt.value is not None
                            default_value = cst.Module([]).code_for_node(stmt.value) if has_default else None

                            self.parameters[param_name] = {
                                "type": param_type,
                                "docstring": docstring,
                                "has_default": has_default,
                                "default_value": default_value,
                            }

    @staticmethod
    def _extract_docstring(
        body_statements: list[cst.CSTNode],
        field_index: int,
    ) -> str:
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


class MethodArgumentsCollector(cst.CSTVisitor):
    """Collects parameter types and docstrings from a method."""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.parameters: dict[str, dict[str, str]] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if node.name.value != self.method_name:
            return
        # Extract docstring
        docstring = self._extract_docstring(node)
        param_docs = self._parse_docstring_params(docstring)
        # Collect parameters
        for param in node.params.params + node.params.kwonly_params:
            if param.name.value == "self" or param.name.value in CORE_PARAMETERS:
                continue
            param_type = cst.Module([]).code_for_node(param.annotation.annotation) if param.annotation else "Any"
            self.parameters[param.name.value] = {"type": param_type, "docstring": param_docs.get(param.name.value, "")}

    def _extract_docstring(self, node: cst.FunctionDef) -> str:
        """Extract docstring from function node."""
        if (
            isinstance(node.body.body[0], cst.SimpleStatementLine)
            and isinstance(node.body.body[0].body[0], cst.Expr)
            and isinstance(node.body.body[0].body[0].value, cst.SimpleString)
        ):
            return node.body.body[0].body[0].value.evaluated_value
        return ""

    def _parse_docstring_params(self, docstring: str) -> dict[str, str]:
        """Parse parameter descriptions from docstring."""
        param_docs = {}
        lines = docstring.split("\n")

        # Find Args section
        args_idx = next((i for i, line in enumerate(lines) if line.strip().lower() == "args:"), None)
        if args_idx is None:
            return param_docs
        # Parse parameter descriptions
        current_param = None
        current_desc = []
        for line in lines[args_idx + 1 :]:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.lower() in ("returns:", "raises:", "example:", "examples:"):
                break

            if stripped_line.endswith(":"):  # Parameter line
                if current_param:
                    param_docs[current_param] = " ".join(current_desc)
                current_desc = []
                # Extract only the parameter name before the first space or parenthesis
                current_param = re.split(r"\s|\(", stripped_line[:-1], 1)[0].strip()
            else:  # Description line
                current_desc.append(stripped_line)
        if current_param:  # Save last parameter
            param_docs[current_param] = " ".join(current_desc)
        return param_docs


#### TREE TRANSFORMERS (UPDATING THE CODE)


class AddImports(cst.CSTTransformer):
    """Transformer that adds import statements to the module."""

    def __init__(self, imports_to_add: list[cst.BaseStatement]):
        self.imports_to_add = imports_to_add
        self.added = False

    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        """Insert the import statements into the module."""
        # If imports were already added, don't add them again
        if self.added:
            return updated_node
        insertion_index = 0
        # Find the index where to insert the imports: make sure the imports are inserted before any code and after all imports (not necessary, we can remove/simplify this part)
        for idx, stmt in enumerate(updated_node.body):
            if not isinstance(stmt, cst.SimpleStatementLine):
                insertion_index = idx
                break
            elif not isinstance(stmt.body[0], (cst.Import, cst.ImportFrom)):
                insertion_index = idx
                break
        # Insert the imports
        new_body = (
            list(updated_node.body[:insertion_index])
            + list(self.imports_to_add)
            + list(updated_node.body[insertion_index:])
        )
        self.added = True
        return updated_node.with_changes(body=new_body)


class UpdateParameters(cst.CSTTransformer):
    """Updates a method's parameters, types, and docstrings."""

    def __init__(self, method_name: str, param_updates: dict[str, dict[str, str]]):
        self.method_name = method_name
        self.param_updates = param_updates
        self.found_method = False  # Flag to check if the method is found

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        # Only proceed if the current function is the target method
        if original_node.name.value != self.method_name:
            return updated_node
        self.found_method = True  # Set the flag as the method is found
        # Update the parameters and docstring of the method
        new_params = self._update_parameters(updated_node.params)
        updated_body = self._update_docstring(updated_node.body)
        # Return the updated function definition
        return updated_node.with_changes(params=new_params, body=updated_body)

    def _update_parameters(self, params: cst.Parameters) -> cst.Parameters:
        """Update parameter types and add new parameters."""
        new_params = list(params.params)  # Copy regular parameters (e.g., 'self')
        new_kwonly_params = []
        # Collect existing parameter names to avoid duplicates
        existing_params = {p.name.value for p in params.params + params.kwonly_params}
        # Update existing keyword-only parameters
        for param in params.kwonly_params:
            param_name = param.name.value
            if param_name in self.param_updates:
                # Update the type annotation for the parameter
                new_annotation = cst.Annotation(
                    annotation=cst.parse_expression(self.param_updates[param_name]["type"])
                )
                new_kwonly_params.append(param.with_changes(annotation=new_annotation))
            else:
                # Keep the parameter as is if no update is needed
                new_kwonly_params.append(param)
        # Add new parameters that are not already present
        for param_name, param_info in self.param_updates.items():
            if param_name not in existing_params:
                # Create a new parameter with the provided type and a default value of None
                annotation = cst.Annotation(annotation=cst.parse_expression(param_info["type"]))
                new_param = cst.Param(
                    name=cst.Name(param_name),
                    annotation=annotation,
                    default=cst.Name(param_info["default_value"]),
                )
                new_kwonly_params.append(new_param)
        # Return the updated parameters object with new and updated parameters
        return params.with_changes(params=new_params, kwonly_params=new_kwonly_params)

    def _update_docstring(self, body: cst.IndentedBlock) -> cst.IndentedBlock:
        """Update parameter descriptions in the docstring."""
        # Check if the first statement is a docstring
        if not (
            isinstance(body.body[0], cst.SimpleStatementLine)
            and isinstance(body.body[0].body[0], cst.Expr)
            and isinstance(body.body[0].body[0].value, cst.SimpleString)
        ):
            # Return the body unchanged if no docstring is found
            return body

        docstring_expr = body.body[0].body[0]
        docstring = docstring_expr.value.evaluated_value  # Get the docstring content
        # Update the docstring content with new and updated parameters
        updated_docstring = self._update_docstring_content(docstring)
        new_docstring = cst.SimpleString(f'"""{updated_docstring}"""')
        # Replace the old docstring with the updated one
        new_body = [body.body[0].with_changes(body=[docstring_expr.with_changes(value=new_docstring)])] + list(
            body.body[1:]
        )
        # Return the updated function body
        return body.with_changes(body=new_body)

    def _update_docstring_content(self, docstring: str) -> str:
        """Update parameter descriptions in the docstring content."""
        # Split parameters into new and updated ones based on their status
        new_params = {name: info for name, info in self.param_updates.items() if info["status"] == "new"}
        update_params = {
            name: info for name, info in self.param_updates.items() if info["status"] in ("update_type", "update_doc")
        }
        # Split the docstring into lines for processing
        docstring_lines = docstring.split("\n")
        # Find or create the "Args:" section and compute indentation levels
        args_index = next((i for i, line in enumerate(docstring_lines) if line.strip().lower() == "args:"), None)
        if args_index is None:
            # If 'Args:' section is not found, insert it before 'Returns:' or at the end
            insertion_index = next(
                (
                    i
                    for i, line in enumerate(docstring_lines)
                    if line.strip().lower() in ("returns:", "raises:", "examples:", "example:")
                ),
                len(docstring_lines),
            )
            docstring_lines.insert(insertion_index, "Args:")
            args_index = insertion_index  # Update the args_index with the new section
        base_indent = docstring_lines[args_index][: -len(docstring_lines[args_index].lstrip())]
        param_indent = base_indent + "    "  # Indentation for parameter lines
        desc_indent = param_indent + "    "  # Indentation for description lines
        # Update existing parameters in the docstring
        if update_params:
            docstring_lines, params_updated = self._process_existing_params(
                docstring_lines, update_params, args_index, param_indent, desc_indent
            )
            # When params_updated is still not empty, it means there are new parameters that are not in the docstring
            # but are in the method signature
            new_params = {**new_params, **params_updated}
        # Add new parameters to the docstring
        if new_params:
            docstring_lines = self._add_new_params(docstring_lines, new_params, args_index, param_indent, desc_indent)
        # Join the docstring lines back into a single string
        return "\n".join(docstring_lines)

    def _format_param_docstring(
        self,
        param_name: str,
        param_info: dict[str, str],
        param_indent: str,
        desc_indent: str,
    ) -> list[str]:
        """Format the docstring lines for a single parameter."""
        # Extract and format the parameter type
        param_type = param_info["type"]
        if param_type.startswith("Optional["):
            param_type = param_type[len("Optional[") : -1]  # Remove Optional[ and closing ]
            optional_str = ", *optional*"
        else:
            optional_str = ""

        # Create the parameter line with type and optionality
        param_line = f"{param_indent}{param_name} (`{param_type}`{optional_str}):"

        # Get and clean up the parameter description
        param_desc = (param_info.get("docstring") or "").strip()
        param_desc = " ".join(param_desc.split())
        if param_desc:
            # Wrap the description text to maintain line width and indentation
            wrapped_desc = textwrap.fill(
                param_desc,
                width=119,
                initial_indent=desc_indent,
                subsequent_indent=desc_indent,
            )
            return [param_line, wrapped_desc]
        else:
            # Return only the parameter line if there's no description
            return [param_line]

    def _process_existing_params(
        self,
        docstring_lines: list[str],
        params_to_update: dict[str, dict[str, str]],
        args_index: int,
        param_indent: str,
        desc_indent: str,
    ) -> tuple[list[str], dict[str, dict[str, str]]]:
        """Update existing parameters in the docstring."""
        # track the params that are updated
        params_updated = params_to_update.copy()
        i = args_index + 1  # Start after the 'Args:' section
        while i < len(docstring_lines):
            line = docstring_lines[i]
            stripped_line = line.strip()
            if not stripped_line:
                # Skip empty lines
                i += 1
                continue
            if stripped_line.lower() in ("returns:", "raises:", "example:", "examples:"):
                # Stop processing if another section starts
                break
            if stripped_line.endswith(":"):
                # Check if the line is a parameter line
                param_line = stripped_line
                param_name = param_line.strip().split()[0]  # Extract parameter name
                if param_name in params_updated:
                    # Get the updated parameter info
                    param_info = params_updated.pop(param_name)
                    # Format the new parameter docstring
                    param_doc_lines = self._format_param_docstring(param_name, param_info, param_indent, desc_indent)
                    # Find the end of the current parameter's description
                    start_idx = i
                    end_idx = i + 1
                    while end_idx < len(docstring_lines):
                        next_line = docstring_lines[end_idx]
                        # Next parameter or section starts or another section starts or empty line
                        if (
                            (next_line.strip().endswith(":") and not next_line.startswith(desc_indent))
                            or next_line.lower() in ("returns:", "raises:", "example:", "examples:")
                            or not next_line
                        ):
                            break
                        end_idx += 1
                    # Insert new param docs and preserve the rest of the docstring
                    docstring_lines = (
                        docstring_lines[:start_idx]  # Keep everything before
                        + param_doc_lines  # Insert new parameter docs
                        + docstring_lines[end_idx:]  # Keep everything after
                    )
                    i = start_idx + len(param_doc_lines)  # Update index to after inserted lines
                i += 1
            else:
                i += 1  # Move to the next line if not a parameter line
        return docstring_lines, params_updated

    def _add_new_params(
        self,
        docstring_lines: list[str],
        new_params: dict[str, dict[str, str]],
        args_index: int,
        param_indent: str,
        desc_indent: str,
    ) -> list[str]:
        """Add new parameters to the docstring."""
        # Find the insertion point after existing parameters
        insertion_index = args_index + 1
        empty_line_index = None
        while insertion_index < len(docstring_lines):
            line = docstring_lines[insertion_index]
            stripped_line = line.strip()
            # Track empty line at the end of Args section
            if not stripped_line:
                if empty_line_index is None:  # Remember first empty line
                    empty_line_index = insertion_index
                insertion_index += 1
                continue
            if stripped_line.lower() in ("returns:", "raises:", "example:", "examples:"):
                break
            empty_line_index = None  # Reset if we find more content
            if stripped_line.endswith(":") and not line.startswith(desc_indent.strip()):
                insertion_index += 1
            else:
                insertion_index += 1

        # If we found an empty line at the end of the Args section, insert before it
        if empty_line_index is not None:
            insertion_index = empty_line_index
        # Prepare the new parameter documentation lines
        param_docs = []
        for param_name, param_info in new_params.items():
            param_doc_lines = self._format_param_docstring(param_name, param_info, param_indent, desc_indent)
            param_docs.extend(param_doc_lines)
        # Insert the new parameters into the docstring
        docstring_lines[insertion_index:insertion_index] = param_docs
        return docstring_lines


#### UTILS


def _check_parameters(
    inference_client_module: cst.Module,
    parameters_module: cst.Module,
    method_name: str,
    parameter_type_name: str,
) -> dict[str, dict[str, Any]]:
    """
    Check for missing parameters and outdated types/docstrings.

    Args:
        inference_client_module: Module containing the InferenceClient
        parameters_module: Module containing the parameters dataclass
        method_name: Name of the method to check
        parameter_type_name: Name of the parameters dataclass

    Returns:
        Dict mapping parameter names to their updates:
        {param_name: {
            "type": str,              # Type annotation
            "docstring": str,         # Parameter documentation
            "status": "new"|"update_type"|"update_doc"  # Whether parameter is new or needs update
        }}
    """
    # Get parameters from the dataclass
    params_collector = DataclassFieldCollector(parameter_type_name)
    parameters_module.visit(params_collector)
    dataclass_params = params_collector.parameters
    # Get existing parameters from the method
    method_collector = MethodArgumentsCollector(method_name)
    inference_client_module.visit(method_collector)
    existing_params = method_collector.parameters

    updates = {}
    # Check for new and updated parameters
    for param_name, param_info in dataclass_params.items():
        if param_name in CORE_PARAMETERS:
            continue
        if param_name not in existing_params:
            # New parameter
            updates[param_name] = {**param_info, "status": "new"}
        else:
            # Check for type/docstring changes
            current = existing_params[param_name]
            normalized_current_doc = _normalize_docstring(current["docstring"])
            normalized_new_doc = _normalize_docstring(param_info["docstring"])
            if current["type"] != param_info["type"]:
                updates[param_name] = {**param_info, "status": "update_type"}
            if normalized_current_doc != normalized_new_doc:
                updates[param_name] = {**param_info, "status": "update_doc"}
    return updates


def _update_parameters(
    module: cst.Module,
    method_name: str,
    param_updates: dict[str, dict[str, str]],
) -> cst.Module:
    """
    Update method parameters, types and docstrings.

    Args:
        module: The module to update
        method_name: Name of the method to update
        param_updates: Dictionary of parameter updates with their type and docstring
            Format: {param_name: {"type": str, "docstring": str, "status": "new"|"update_type"|"update_doc"}}

    Returns:
        Updated module
    """
    transformer = UpdateParameters(method_name, param_updates)
    return module.visit(transformer)


def _get_imports_to_add(
    parameters: dict[str, dict[str, str]],
    parameters_module: cst.Module,
    inference_client_module: cst.Module,
) -> dict[str, list[str]]:
    """
    Get the needed imports for missing parameters.

    Args:
        parameters (dict[str, dict[str, str]]): Dictionary of parameters with their type and docstring.
        eg: {"function_to_apply": {"type": "ClassificationOutputTransform", "docstring": "Function to apply to the input."}}
        parameters_module (cst.Module): The module where the parameters are defined.
        inference_client_module (cst.Module): The module of the inference client.

    Returns:
        dict[str, list[str]]: A dictionary mapping modules to list of types to import.
        eg: {"huggingface_hub.inference._generated.types": ["ClassificationOutputTransform"]}
    """
    # Collect all type names from parameter annotations
    types_to_import = set()
    for param_info in parameters.values():
        types_to_import.update(_collect_type_hints_from_annotation(param_info["type"]))
    # Gather existing imports in the inference client module
    context = CodemodContext()
    gather_visitor = GatherImportsVisitor(context)
    inference_client_module.visit(gather_visitor)
    # Map types to their defining modules in the parameters module
    module_collector = ModulesCollector()
    parameters_module.visit(module_collector)
    # Determine which imports are needed

    needed_imports = {}
    for type_name in types_to_import:
        types_to_modules = module_collector.type_to_module
        module = types_to_modules.get(type_name, DEFAULT_MODULE)
        # Maybe no need to check that since the code formatter will handle duplicate imports?
        if module not in gather_visitor.object_mapping or type_name not in gather_visitor.object_mapping[module]:
            needed_imports.setdefault(module, []).append(type_name)
    return needed_imports


def _generate_import_statements(import_dict: dict[str, list[str]]) -> str:
    """
    Generate import statements from a dictionary of needed imports.

    Args:
        import_dict (dict[str, list[str]]): Dictionary mapping modules to list of types to import.
        eg: {"typing": ["List", "Dict"], "huggingface_hub.inference._generated.types": ["ClassificationOutputTransform"]}

    Returns:
        str: The import statements as a string.
    """
    import_statements = []
    for module, imports in import_dict.items():
        if imports:
            import_list = ", ".join(imports)
            import_statements.append(f"from {module} import {import_list}")
        else:
            import_statements.append(f"import {module}")
    return "\n".join(import_statements)


def _normalize_docstring(docstring: str) -> str:
    """Normalize a docstring by removing extra whitespace, newlines and indentation."""
    # Split into lines, strip whitespace from each line, and join back
    return " ".join(line.strip() for line in docstring.split("\n")).strip()


# TODO: Needs to be improved, maybe using `typing.get_type_hints` instead (we gonna need to access the method though)?
def _collect_type_hints_from_annotation(annotation_str: str) -> set[str]:
    """
    Collect type hints from an annotation string.

    Args:
        annotation_str (str): The annotation string.

    Returns:
        set[str]: A set of type hints.
    """
    type_string = annotation_str.replace(" ", "")
    builtin_types = {d for d in dir(builtins) if isinstance(getattr(builtins, d), type)}
    types = re.findall(r"\w+|'[^']+'|\"[^\"]+\"", type_string)
    extracted_types = {t.strip("\"'") for t in types if t.strip("\"'") not in builtin_types}
    return extracted_types


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


def _check_and_update_parameters(
    method_params: dict[str, str],
    update: bool,
) -> NoReturn:
    """
    Check if task methods have missing parameters and update the InferenceClient source code if needed.
    """
    merged_imports = defaultdict(set)
    logs = []
    inference_client_filename = INFERENCE_CLIENT_FILE
    # Read and parse the inference client module
    inference_client_module = _parse_module_from_file(inference_client_filename)
    modified_module = inference_client_module
    has_changes = False

    for method_name, parameter_type_name in method_params.items():
        parameters_filename = INFERENCE_TYPES_PATH / f"{method_name}.py"
        parameters_module = _parse_module_from_file(parameters_filename)

        # Check for missing parameters
        updates = _check_parameters(
            modified_module,
            parameters_module,
            method_name,
            parameter_type_name,
        )

        if not updates:
            continue

        if update:
            ## Get missing imports to add
            needed_imports = _get_imports_to_add(updates, parameters_module, modified_module)
            for module, imports_to_add in needed_imports.items():
                merged_imports[module].update(imports_to_add)
            modified_module = _update_parameters(modified_module, method_name, updates)
            has_changes = True
        else:
            logs.append(f"\n🔧 Updates needed in method `{method_name}`:")
            new_params = [p for p, i in updates.items() if i["status"] == "new"]
            updated_params = {
                p: "type" if i["status"] == "update_type" else "docstring"
                for p, i in updates.items()
                if i["status"] in ("update_type", "update_doc")
            }
            if new_params:
                for param in sorted(new_params):
                    logs.append(f"   • {param} (missing)")

            if updated_params:
                for param, update_type in sorted(updated_params.items()):
                    logs.append(f"   • {param} (outdated {update_type})")

    if has_changes:
        if merged_imports:
            import_statements = _generate_import_statements(merged_imports)
            imports_to_add = cst.parse_module(import_statements).body
            # Update inference client module with the missing imports
            modified_module = modified_module.visit(AddImports(imports_to_add))
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
            "Please run `make inference_update` or `python utils/check_task_parameters.py --update"
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


def update_inference_client(update: bool):
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
    _check_and_update_parameters(method_params, update=update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help=("Whether to update `./src/huggingface_hub/inference/_client.py` if parameters are missing."),
    )
    args = parser.parse_args()
    update_inference_client(update=args.update)
