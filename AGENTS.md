# Agent Guide for huggingface_hub

## Project overview

Python client library for the Hugging Face Hub. Source code is in `src/huggingface_hub/`, tests in `tests/`.

## Setup

- **Virtualenv**: `.venv` (everything pre-installed in editable mode with dev extras).
- **Activate**: `source .venv/bin/activate` (or prefix commands with `.venv/bin/python`).

## Key commands

| Command                                        | What it does                                                                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `make style`                                   | Auto-format code (ruff format + fix), update generated files (static imports, `__all__`, async client, CLI reference) |
| `make quality`                                 | Check formatting, linting, generated files, and type-check (`ty check src`)                                           |
| `pytest tests/test_<module>.py`                | Run a specific test file                                                                                              |
| `pytest tests/test_<module>.py -k "test_name"` | Run a single test                                                                                                     |
| `pytest tests/`                                | Run all tests (slow, many require network/auth)                                                                       |

**Always run `make style` then `make quality` before committing.**

## Code structure

### Core modules (`src/huggingface_hub/`)

- `__init__.py` — Public API surface. Static imports are **auto-generated** by `utils/check_static_imports.py`; run `make style` to update.
- `hf_api.py` — Main `HfApi` class (~11k lines). Repo CRUD, uploads, downloads, discussions, PRs, collections, and more. Most Hub operations live here.
- `file_download.py` — `hf_hub_download`, caching logic, ETag/metadata resolution, xet download support.
- `_snapshot_download.py` — `snapshot_download` (download an entire repo).
- `_commit_api.py` — Low-level commit operations (`CommitOperationAdd`, `CommitOperationDelete`, `CommitOperationCopy`), LFS upload handling.
- `_commit_scheduler.py` — Background scheduled commits.
- `_upload_large_folder.py` — Chunked upload for very large folders.
- `hf_file_system.py` — `HfFileSystem` (fsspec-based POSIX-like filesystem for Hub repos).
- `hub_mixin.py` — `ModelHubMixin` base class for ML framework integration (save/load to Hub).
- `repocard.py` / `repocard_data.py` — `RepoCard`, `ModelCard`, `DatasetCard` and their metadata.
- `community.py` — `Discussion`, `DiscussionComment` and event deserialization.
- `lfs.py` — Git LFS batch upload utilities.
- `_login.py` — `login()`, `logout()`, `notebook_login()`, token management.
- `_inference_endpoints.py` — Inference endpoint CRUD and scaling.
- `_jobs_api.py` — Training jobs API.
- `_space_api.py` — Space runtime/hardware/storage types.
- `_webhooks_server.py` / `_webhooks_payload.py` — Webhook server and payload definitions.
- `constants.py` — All environment-variable-driven constants (timeouts, cache paths, endpoints).
- `errors.py` — Custom exception hierarchy (`HfHubHTTPError`, `RepositoryNotFoundError`, etc.).

### Inference (`src/huggingface_hub/inference/`)

- `_client.py` — `InferenceClient` (very large, 40+ task methods). Async version is **auto-generated** by `utils/generate_async_inference_client.py`.
- `_generated/` — Auto-generated type definitions for all inference tasks. **Do not edit by hand**; regenerate with `utils/generate_inference_types.py`.
- `_providers/` — Provider adapters (24+): `hf_inference`, `openai`, `cohere`, `together`, `fireworks`, `cerebras`, `fal`, `replicate`, etc. Each maps the unified API to a provider's HTTP API.
- `_mcp/` — Model Context Protocol integration (agent, client, CLI).

### Utilities (`src/huggingface_hub/utils/`)

- `_http.py` — HTTP client factory, `http_backoff` / `http_stream_backoff` retry helpers, header building.
- `_auth.py` — Token retrieval from env/files/credential helpers.
- `_cache_manager.py` — Cache browsing and deletion (`scan-cache`, `delete-cache` CLI).
- `_validators.py` — Input validation decorators (`@validate_hf_hub_args`).
- `_runtime.py` — Runtime detection (is torch/tf/xet/etc. available?).
- `_deprecation.py` — `@_deprecate_arguments`, `@_deprecate_method` decorators.
- `_headers.py` — `build_hf_headers` (user-agent, auth).
- `_pagination.py` — Paginated API response helpers.
- `tqdm.py` — Custom progress bar wrappers.
- Other: `_datetime.py`, `_dotenv.py`, `_git_credential.py`, `_subprocess.py`, `_telemetry.py`, `sha.py`, `_xet.py`, ...

### CLI (`src/huggingface_hub/cli/`)

Entry point: `hf.py` (Typer app). Subcommands split into modules: `auth.py`, `repos.py`, `spaces.py`, `models.py`, `datasets.py`, `buckets.py`, `jobs.py`, `collections.py`, `webhooks.py`, etc.

#### Adding CLI commands

**Structure & naming conventions:**

- Commands are organized as **Typer groups** registered in `hf.py` (e.g. `app.add_typer(spaces_cli, name="spaces")`).
- Each module creates its app with `typer_factory(help="...")` and defines commands with `@app.command("name")`.
- Use **pipe-separated aliases**: `@app.command("list | ls")` registers both `list` and `ls`.
- Use standard **verb names**: `ls`/`list`, `info`, `create`, `set`, `delete`, `update`.
- For **sub-resources** with multiple operations, create a nested subgroup: `volumes_cli = typer_factory(...)` then `spaces_cli.add_typer(volumes_cli, name="volumes")` → gives `hf spaces volumes ls/set/delete`. See `repos.py` for examples (`tag_cli`, `branch_cli`).

**Reusable option types** (from `_cli_utils.py` — import and use these, don't reinvent):

- `TokenOpt`, `RepoIdArg`, `RepoTypeOpt`, `RevisionOpt` — standard auth/repo options.
- `SearchOpt`, `AuthorOpt`, `FilterOpt`, `LimitOpt` — list/search options.
- `FormatWithAutoOpt` / `FormatOpt` — output format (`auto|json|human|quiet|agent`).
- `VolumesOpt` + `parse_volumes()` — `-v`/`--volume` flag with `hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]` syntax.
- `get_hf_api(token=token)` — creates an `HfApi` instance with token.
- `api_object_to_dict(obj)` — converts dataclass API objects to dicts for output.

**Output** (from `_output.py` — use the `out` singleton):

- `out.table(items)` — list results (auto-formats as padded table / TSV / JSON depending on `--format`).
- `out.dict(data)` — single-item detail view.
- `out.result("Message", key=value, ...)` — success summary with green checkmark.
- `out.confirm("Prompt?", yes=yes)` — confirmation for destructive operations. Pair with a `-y`/`--yes` flag.
- `out.hint("...")` — actionable follow-up suggestion. Try to add hints when adding new commands or refactoring a command. Hints should preferably reuse the input args to be specific to the current use case. Example: `out.hint(f"Use 'hf buckets ls {bucket_id}' to list files from the bucket.")` after a bucket creation.
- `out.text()`, `out.warning()`, `out.error()` — free-form output.

**Destructive operations** should use `out.confirm()` with a `yes: Annotated[bool, typer.Option("-y", "--yes", help="Answer Yes to prompt automatically.")]` parameter.

**Errors**: raise `CLIError("message")` for user-facing errors. Never wrap API calls with try/except for `RepositoryNotFoundError`, `RevisionNotFoundError`, etc. (already done globally)

**Generated docs**: `make style` auto-regenerates `docs/source/en/package_reference/cli.md` via `utils/generate_cli_reference.py`. Don't edit that file by hand.

**Guides**: update the CLI guide `docs/sources/en/guides/cli.md` when adding / updating CLI commands. If the command is specific to a topic which has its own guide in `docs/sources/en/guides`, add a mention in the guide as well, using the same tone as the existing guide.

**CLI tests**: add tests in `tests/test_cli.py`. Try to group tests into classes when relevant. Do not add a test for each specific use case / parameter set. Usually testing the 1-2 main use cases is enough.


### Type checking: local vs CI

Locally, `make quality` runs `ty check src` using whatever version of `ty` is installed in the virtualenv. CI (`.github/workflows/python-quality.yml`) runs `uvx ty check src` (always the latest `ty` release) **and** `mypy src`. This means CI may flag errors that do not appear locally — this is intentional.

- **`ty`**: The local version is whatever the developer has installed — it won't update on its own, so new `ty` releases don't unexpectedly break the workflow. Developers can update it at will. CI always uses the latest version to catch issues early.
- **`mypy`**: Historically the project's type checker. It is kept in CI (until `ty` reaches a stable release) but not run locally via `make quality` because it takes 1–2 minutes, which slows down the development loop.

If CI fails on type checks that pass locally, the likely cause is a newer `ty` version or a `mypy`-only diagnostic. Fix the reported errors rather than downgrading the checker.

## Commits & PRs

- **Commit message prefix**: use `[Area]` prefix matching the scope, e.g. `[CLI] Add ...`, `[CLI] Fix ...`, `[Inference] ...`.
- **PR title**: short (under 70 chars), same `[Area]` prefix convention.
- **PR description**: keep it casual. Include a `## Summary` with a few bullet points and real CLI/code **examples** from manual testing (copy-paste terminal output). No need for a formal "Test plan" section.
