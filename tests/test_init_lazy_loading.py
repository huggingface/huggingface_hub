import unittest

import jedi


class TestHuggingfaceHubInit(unittest.TestCase):
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
                self.assertEqual(completion.module_name, "huggingface_hub")

                # Assert autocomplete knows where `create_commit` lives
                # It would not be the case with a dynamic import.
                goto_list = completion.goto()
                self.assertEqual(len(goto_list), 1)

                # Assert docstring is find. This means autocomplete can also provide
                # the help section.
                signature_list = goto_list[0].get_signatures()
                self.assertEqual(len(signature_list), 2)  # create_commit has 2 signatures (normal and `run_as_future`)
                self.assertTrue(signature_list[0].docstring().startswith("create_commit(repo_id: str,"))
                break
        else:
            self.fail(
                "Jedi autocomplete did not suggest `create_commit` to complete the"
                f" line `{source}`. It is most probable that static imports are not"
                " correct in `./src/huggingface_hub/__init__.py`. Please run `make"
                " style` to fix this."
            )
