import subprocess
import sys

import jedi
import pytest


class TestHuggingfaceHubInit:
    def test_utils_are_lazy_loaded(self) -> None:
        script = """
import sys
import huggingface_hub.utils as utils
assert "huggingface_hub.utils._cache_manager" not in sys.modules
utils.scan_cache_dir
assert "huggingface_hub.utils._cache_manager" in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_cli_commands_are_lazy_loaded(self) -> None:
        script = """
import sys
from click.testing import CliRunner
from huggingface_hub.cli.hf import app
assert "huggingface_hub.cli.models" not in sys.modules
result = CliRunner().invoke(app, ["models", "--help"])
assert result.exit_code == 0, result.output
assert "huggingface_hub.cli.models" in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_hf_api_dependencies_are_lazy_loaded(self) -> None:
        script = """
import sys
from huggingface_hub import HfApi
assert HfApi.__name__ == "HfApi"
assert "huggingface_hub._commit_api" not in sys.modules
assert "huggingface_hub.file_download" not in sys.modules
assert "huggingface_hub._dataset_viewer" not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_star_import_does_not_load_optional_frameworks(self) -> None:
        script = """
import sys
from huggingface_hub import *
for module in ("fastapi", "numpy", "starlette", "torch"):
    assert module not in sys.modules, module
assert HFSummaryWriter.__name__ == "HFSummaryWriter"
assert WebhooksServer.__name__ == "WebhooksServer"
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    @pytest.mark.skip(
        reason="`jedi.Completion.get_signatures()` output differs between Python 3.12 and earlier versions, affecting test consistency"
    )
    def test_autocomplete_on_root_imports(self) -> None:
        """Test autocomplete with `huggingface_hub` works with Jedi.

        Not all autocomplete systems are based on Jedi but if this one works we can
        assume others do as well.
        """
        source = """from huggingface_hub import c"""
        script = jedi.Script(source, path="example.py")
        completions = script.complete(1, len(source))

        for completion in completions:
            if completion.name == "create_commit":
                # Assert `create_commit` is suggestion from `huggingface_hub` lib
                assert completion.module_name == "huggingface_hub"

                # Assert autocomplete knows where `create_commit` lives
                # It would not be the case with a dynamic import.
                goto_list = completion.goto()
                assert len(goto_list) == 1

                # Assert docstring is find. This means autocomplete can also provide
                # the help section.
                signature_list = goto_list[0].get_signatures()
                assert len(signature_list) == 2  # create_commit has 2 signatures (normal and `run_as_future`)
                assert signature_list[0].docstring().startswith("create_commit(repo_id: str,")
                break
        else:
            pytest.fail(
                "Jedi autocomplete did not suggest `create_commit` to complete the"
                f" line `{source}`. It is most probable that static imports are not"
                " correct in `./src/huggingface_hub/__init__.py`. Please run `make"
                " style` to fix this."
            )
