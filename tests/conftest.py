from pytest import Parser


def pytest_addoption(parser: Parser) -> None:
    """Add option to update `huggingface_hub.__init__.py` with new imports.

    If the init file is updated, the test itself will fail.
    See `./tests/test_init_lazy_loading.py` for more details.

    Run the following command to update static imports:
    ```
    pytest tests/test_init_lazy_loading.py -k test_static_imports --update-init-file
    ```
    """
    parser.addoption(
        "--update-init-file",
        action="store_true",
        help=(
            "If True, the root `__init__` file will be updated with new sorted imports."
        ),
    )
