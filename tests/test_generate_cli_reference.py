import click

from utils.generate_cli_reference import _normalize_command_aliases, get_docs_for_click


def test_normalize_nested_aliases_in_usage_lines() -> None:
    content = """
#### `hf repos | repo tag list | ls`

List tags for a repo.

**Usage**:

```console
$ hf repos | repo tag list | ls [OPTIONS] REPO_ID
```
"""

    normalized = _normalize_command_aliases(content)

    assert "#### `hf repos tag list`" in normalized
    assert "List tags for a repo. [alias: ls]" in normalized
    assert "$ hf repos tag list [OPTIONS] REPO_ID" in normalized
    assert "$ hf repos | repo tag list [OPTIONS] REPO_ID" not in normalized


def test_normalize_parent_and_child_aliases_in_usage_lines() -> None:
    content = """
### `hf extensions | ext list | ls`

List installed extensions.

**Usage**:

```console
$ hf extensions | ext list | ls [OPTIONS]
```
"""

    normalized = _normalize_command_aliases(content)

    assert "$ hf extensions list [OPTIONS]" in normalized
    assert "$ hf extensions | ext list [OPTIONS]" not in normalized


def test_short_help_is_not_truncated() -> None:
    @click.group()
    def cli() -> None: ...

    @cli.command()
    def verify() -> None:
        """Verify checksums for a single repo revision from cache or a local directory.

        More details here.
        """

    docs = get_docs_for_click(obj=cli, ctx=click.Context(cli, info_name="hf"), name="hf")

    assert "* `verify`: Verify checksums for a single repo revision from cache or a local directory.\n" in docs
