"""
Utility script to check and update the InferenceClient task methods arguments and docstrings
based on the tasks input parameters.

What this script does:
- [x] detect missing parameters in method signature
- [x] add missing parameters to methods signature
- [x] detect missing parameters in method docstrings
- [x] add missing parameters to methods docstrings
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
from typing import Dict, List, NoReturn, Optional, Set

import libcst as cst
from helpers import format_source_code
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import GatherImportsVisitor

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


#### TREE TRANSFORMERS (UPDATING THE CODE)


class AddParameters(cst.CSTTransformer):
    """Updates a method by adding missing parameters and updating the docstring."""

    def __init__(self, method_name: str, missing_params: Dict[str, Dict[str, str]]):
        self.method_name = method_name
        self.missing_params = missing_params
        self.found_method = False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == self.method_name:
            self.found_method = True
            new_params = self._update_parameters(updated_node.params)
            updated_body = self._update_docstring(updated_node.body)
            return updated_node.with_changes(params=new_params, body=updated_body)
        return updated_node

    def _update_parameters(self, params: cst.Parameters) -> cst.Parameters:
        new_kwonly_params = list(params.kwonly_params)
        existing_args = {param.name.value for param in params.params + params.kwonly_params}

        for param_name, param_info in self.missing_params.items():
            if param_name not in existing_args:
                annotation = cst.Annotation(annotation=cst.parse_expression(param_info["type"]))
                new_param = cst.Param(
                    name=cst.Name(param_name),
                    annotation=annotation,
                    default=cst.Name("None"),
                )
                new_kwonly_params.append(new_param)

        return params.with_changes(kwonly_params=new_kwonly_params)

    def _update_docstring(self, body: cst.IndentedBlock) -> cst.IndentedBlock:
        if not isinstance(body.body[0], cst.SimpleStatementLine) or not isinstance(body.body[0].body[0], cst.Expr):
            return body

        docstring_expr = body.body[0].body[0]
        if not isinstance(docstring_expr.value, cst.SimpleString):
            return body

        docstring = docstring_expr.value.evaluated_value
        updated_docstring = self._update_docstring_content(docstring)
        new_docstring = cst.SimpleString(f'"""{updated_docstring}"""')
        new_body = [body.body[0].with_changes(body=[docstring_expr.with_changes(value=new_docstring)])] + list(
            body.body[1:]
        )
        return body.with_changes(body=new_body)

    def _update_docstring_content(self, docstring: str) -> str:
        docstring_lines = docstring.split("\n")

        # Step 1: find the right insertion index
        args_index = next((i for i, line in enumerate(docstring_lines) if line.strip().lower() == "args:"), None)
        # If there is no "Args:" section, insert it after the first section that is not empty and not a sub-section
        if args_index is None:
            insertion_index = next(
                (
                    i
                    for i, line in enumerate(docstring_lines)
                    if line.strip().lower() in ("returns:", "raises:", "examples:", "example:")
                ),
                len(docstring_lines),
            )
            docstring_lines.insert(insertion_index, "Args:")
            args_index = insertion_index
            insertion_index += 1
        else:
            # Find the next section (in this order: Returns, Raises, Example(s))
            next_section_index = next(
                (
                    i
                    for i, line in enumerate(docstring_lines)
                    if line.strip().lower() in ("returns:", "raises:", "example:", "examples:")
                ),
                None,
            )
            if next_section_index is not None:
                # If there's a blank line before "Returns:", insert before that blank line
                if next_section_index > 0 and docstring_lines[next_section_index - 1].strip() == "":
                    insertion_index = next_section_index - 1
                else:
                    # If there's no blank line, insert at the "Returns:" line and add a blank line after insertion
                    insertion_index = next_section_index
                    docstring_lines.insert(insertion_index, "")
            else:
                # If there's no next section, insert at the end
                insertion_index = len(docstring_lines)

        # Step 2: format the parameter docstring
        # Calculate the base indentation
        base_indentation = docstring_lines[args_index][
            : len(docstring_lines[args_index]) - len(docstring_lines[args_index].lstrip())
        ]
        param_indentation = base_indentation + "    "  # Indent parameters under "Args:"
        description_indentation = param_indentation + "    "  # Indent descriptions under parameter names

        param_docs = []
        for param_name, param_info in self.missing_params.items():
            param_type_str = param_info["type"].replace("Optional[", "").rstrip("]")
            optional_str = "*optional*" if "Optional[" in param_info["type"] else ""
            param_docstring = (param_info.get("docstring") or "").strip()

            # Clean up the docstring to remove extra spaces
            param_docstring = " ".join(param_docstring.split())

            # Prepare the parameter line
            param_line = f"{param_indentation}{param_name} (`{param_type_str}`, {optional_str}):"

            # Wrap the parameter docstring
            wrapped_description = textwrap.fill(
                param_docstring,
                width=119,
                initial_indent=description_indentation,
                subsequent_indent=description_indentation,
            )

            # Combine parameter line and description
            if param_docstring:
                param_doc = f"{param_line}\n{wrapped_description}"
            else:
                param_doc = param_line

            param_docs.append(param_doc)

        # Step 3: insert the new parameter docs into the docstring
        docstring_lines[insertion_index:insertion_index] = param_docs
        return "\n".join(docstring_lines)


class AddImports(cst.CSTTransformer):
    """Transformer that adds import statements to the module."""

    def __init__(self, imports_to_add: List[cst.BaseStatement]):
        self.imports_to_add = imports_to_add
        self.added = False

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
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


#### UTILS


def check_missing_parameters(
    inference_client_module: cst.Module,
    parameters_module: cst.Module,
    method_name: str,
    parameter_type_name: str,
) -> Dict[str, Dict[str, str]]:
    # Get parameters from the parameters module
    params_collector = DataclassFieldCollector(parameter_type_name)
    parameters_module.visit(params_collector)
    parameters = params_collector.parameters

    # Get existing arguments from the method
    method_argument_collector = ArgumentsCollector(method_name)
    inference_client_module.visit(method_argument_collector)
    existing_args = method_argument_collector.existing_args
    missing_params = {k: v for k, v in parameters.items() if k not in existing_args}
    return missing_params


def update_missing_parameters(
    module: cst.Module,
    method_name: str,
    missing_params: Dict[str, Dict[str, str]],
) -> cst.Module:
    """Update the method signature by adding missing parameters."""
    transformer = AddParameters(method_name, missing_params)
    return module.visit(transformer)


def get_imports_to_add(
    parameters: Dict[str, Dict[str, str]],
    parameters_module: cst.Module,
    inference_client_module: cst.Module,
) -> Dict[str, List[str]]:
    """
    Get the needed imports for missing parameters.

    Args:
        parameters (Dict[str, Dict[str, str]]): Dictionary of parameters with their type and docstring.
        eg: {"function_to_apply": {"type": "ClassificationOutputTransform", "docstring": "Function to apply to the input."}}
        parameters_module (cst.Module): The module where the parameters are defined.
        inference_client_module (cst.Module): The module of the inference client.

    Returns:
        Dict[str, List[str]]: A dictionary mapping modules to list of types to import.
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


def _generate_import_statements(import_dict: Dict[str, List[str]]) -> str:
    """
    Generate import statements from a dictionary of needed imports.

    Args:
        import_dict (Dict[str, List[str]]): Dictionary mapping modules to list of types to import.
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


# TODO: Needs to be improved, maybe using `typing.get_type_hints` instead (we gonna need to access the method though)?
def _collect_type_hints_from_annotation(annotation_str: str) -> Set[str]:
    """
    Collect type hints from an annotation string.

    Args:
        annotation_str (str): The annotation string.

    Returns:
        Set[str]: A set of type hints.
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


def _check_parameters(
    method_params: Dict[str, str],
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
        missing_params = check_missing_parameters(modified_module, parameters_module, method_name, parameter_type_name)

        if not missing_params:
            continue

        if update:
            if missing_params:
                # Handle missing parameters (existing code)
                needed_imports = get_imports_to_add(missing_params, parameters_module, modified_module)
                for module, imports_to_add in needed_imports.items():
                    merged_imports[module].update(imports_to_add)
                modified_module = update_missing_parameters(modified_module, method_name, missing_params)

            has_changes = True
        else:
            if missing_params:
                logs.append(f"❌ Missing parameters found in `{method_name}`.")

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
    _check_parameters(method_params, update=update)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help=("Whether to update `./src/huggingface_hub/inference/_client.py` if parameters are missing."),
    )
    args = parser.parse_args()
    update_inference_client(update=args.update)
