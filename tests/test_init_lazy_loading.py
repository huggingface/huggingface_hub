import unittest
from pathlib import Path

import pytest

import huggingface_hub
import isort
import jedi
from huggingface_hub import _SUBMOD_ATTRS  # which modules/functions are available ?


IF_TYPE_CHECKING_LINE = "\nif TYPE_CHECKING:\n"
SETUP_CFG_PATH = Path(__file__).parent.parent / "setup.cfg"


@pytest.fixture
def update_init_file(request):
    request.cls.update_init_file = request.config.getoption("--update-init-file")


@pytest.mark.usefixtures("update_init_file")
class TestHuggingfaceHubInit(unittest.TestCase):
    update_init_file: bool

    def test_static_imports(self) -> None:
        """Test all imports are made twice (1 in lazy-loading and 1 in static checks).

        For more explanations, see `./src/huggingface_hub/__init__.py`.
        Run the following command to update static imports.
        ```
        pytest tests/test_init_lazy_loading.py -k test_static_imports --update-init-file
        ```
        """
        init_path = Path(huggingface_hub.__file__)
        with init_path.open() as f:
            init_content = f.read()

        init_content_before_static_checks = init_content.split(IF_TYPE_CHECKING_LINE)[0]

        static_imports = [
            f"    from .{module} import {attr} # noqa: F401"
            for module, attributes in _SUBMOD_ATTRS.items()
            for attr in attributes
        ]

        expected_init_content_raw = (
            init_content_before_static_checks
            + IF_TYPE_CHECKING_LINE
            + "\n".join(static_imports)
            + "\n"
        )

        expected_init_content_cleaned = isort.code(
            expected_init_content_raw, config=isort.Config(settings_path=SETUP_CFG_PATH)
        )

        if init_content != expected_init_content_cleaned:
            if self.update_init_file:
                with init_path.open("w") as f:
                    f.write(expected_init_content_cleaned)

                self.fail(
                    "Pytest was run with '--update-init-file' option and"
                    " `./src/huggingface_hub/__init__.py` content has been updated. It"
                    " is most likely that you added a module/function to"
                    " `_SUBMOD_ATTRS` and did not update the 'static import'-part."
                    " Please make sure the changes are accurate and if yes, commit them"
                    " and re-run this test without the '--update-init-file' option."
                )
            else:
                self.fail(
                    "Expected content mismatch in `./src/huggingface_hub/__init__.py`."
                    " It is most likely that you added a module/function to"
                    " `_SUBMOD_ATTRS` and did not update the 'static import'-part. To"
                    " do it, please re-run the test suite with '--update-init-file'"
                    " option and look at the changes. If the changes are accurate,"
                    " commit them and re-run this test without the '--update-init-file'"
                    " option."
                )

        self.assertEqual(init_content, expected_init_content_cleaned)

    def test_autocomplete_on_root_imports(self) -> None:
        """Test autocomplete with `huggingface_hub` works with Jedi.

        Not all autocomplete systems are based on Jedi but if this one works we can
        assume others does it as well.
        """
        source = """from huggingface_hub import c"""
        script = jedi.Script(source, path="example.py")
        completions = script.complete(1, len(source))

        for completion in completions:
            if completion.name == "create_commit":
                # Assert `create_commit` is suggestion from `huggingface_hub` lib
                self.assertEquals(completion.module_name, "huggingface_hub")

                # Assert autocomplete knows where `create_commit` lives
                # It would not be the case with a dynamic import.
                goto_list = completion.goto()
                self.assertEquals(len(goto_list), 1)

                # Assert docstring is find. This means autocomplete can also provide
                # the help section.
                signature_list = goto_list[0].get_signatures()
                self.assertEquals(len(signature_list), 1)
                self.assertTrue(
                    signature_list[0]
                    .docstring()
                    .startswith("create_commit(self, repo_id: str")
                )
                break
        else:
            self.fail(
                "Jedi autocomplete did not suggest `create_commit` to complete the"
                f" line `{source}`. It is most probable that static imports are not"
                " correct in `./src/huggingface_hub/__init__.py`. Please run `pytest"
                " tests/test_init_lazy_loading.py -k test_static_imports"
                " --update-init-file` to fix this."
            )
