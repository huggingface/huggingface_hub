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

### Serialization (`src/huggingface_hub/serialization/`)

- `_torch.py` — `save_torch_model()`, `load_torch_model()`, state-dict splitting, safetensors support.
- `_dduf.py` — DDUF format support.

### CLI (`src/huggingface_hub/cli/`)

Entry point: `hf.py` (Typer app). Subcommands split into modules: `auth.py`, `repo.py`, `repo_files.py`, etc.

### Tests (`tests/`)

- One `test_<module>.py` per source module (e.g. `test_hf_api.py`, `test_file_download.py`, `test_inference_client.py`).
- `conftest.py` — Fixtures (temp cache dirs, env patching).
- `testing_utils.py` / `testing_constants.py` — Shared test helpers and staging-repo constants.
- `cassettes/` — Recorded HTTP responses for offline tests (`@pytest.mark.vcr`). **Do not add new cassettes.**
- `fixtures/` — Static test data.

### Dev scripts (`utils/`)

- `check_static_imports.py` — Ensures `__init__.py` static imports match the lazy loader.
- `check_all_variable.py` — Validates `__all__` exports.
- `generate_async_inference_client.py` — Generates `AsyncInferenceClient` from sync client.
- `generate_inference_types.py` — Generates inference type definitions.
- `generate_cli_reference.py` — Generates CLI docs.

## Style

- Max line length: 119 chars.
- Linter/formatter: `ruff`.
- Imports sorted by `ruff` (isort-compatible).

### Type checking: local vs CI

Locally, `make quality` runs `ty check src` using whatever version of `ty` is installed in the virtualenv. CI (`.github/workflows/python-quality.yml`) runs `uvx ty check src` (always the latest `ty` release) **and** `mypy src`. This means CI may flag errors that do not appear locally — this is intentional.

- **`ty`**: The local version is whatever the developer has installed — it won't update on its own, so new `ty` releases don't unexpectedly break the workflow. Developers can update it at will. CI always uses the latest version to catch issues early.
- **`mypy`**: Historically the project's type checker. It is kept in CI (until `ty` reaches a stable release) but not run locally via `make quality` because it takes 1–2 minutes, which slows down the development loop.

If CI fails on type checks that pass locally, the likely cause is a newer `ty` version or a `mypy`-only diagnostic. Fix the reported errors rather than downgrading the checker.

## Testing notes

- Tests use `pytest` with `pytest-env` setting `HUGGINGFACE_CO_STAGING=1` (tests hit staging Hub by default).
- Most integration tests require `HF_TOKEN` to be set. Unit tests don't.
- do not register or commit new HTTP cassettes.
