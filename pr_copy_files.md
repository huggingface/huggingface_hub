## Summary

Add a new `HfApi.copy_files` API and wire `hf buckets cp` to support remote HF-handle copy flows.

### Implemented

- Added `HfApi.copy_files(source, destination, *, token=None) -> int`.
- Supported copy directions:
  - bucket -> same bucket
  - bucket -> different bucket
  - repo (model/dataset/space) -> bucket
- Explicitly reject bucket/repo destinations that are not buckets (bucket -> repo and repo -> repo).
- Added folder-copy rule: destination must end with `/` for folder sources.
- Added repo `@revision` support in source handles (including special refs parsing).
- Added top-level alias export: `copy_files = api.copy_files`.

## Implementation details

- Source/destination are parsed as HF handles (`hf://...`) with dedicated copy-handle parsing.
- For repo sources:
  - if `xet_hash` exists: direct bucket add by hash
  - if `xet_hash is None`: download file then re-upload to bucket
- For bucket sources:
  - same-bucket copy: direct add by hash
  - cross-bucket copy: download + re-upload fallback (required by backend behavior)
- Extended internal `_batch_bucket_files` to accept prebuilt `_BucketAddFile` entries and skip Xet upload when `xet_hash` is already known.

## CLI changes

- `hf buckets cp` now supports remote-to-remote handle copies by delegating to `api.copy_files`.
- Added new examples for:
  - bucket -> bucket
  - repo -> bucket
- Output now reports count for remote copies:
  - `Copied <N> file(s): SRC -> DST`

## Tests

### Added in `tests/test_buckets.py`

- `test_copy_files_bucket_to_same_bucket_file`
- `test_copy_files_bucket_to_different_bucket_folder`
- `test_copy_files_repo_to_bucket_with_revision`
- `test_copy_files_bucket_to_repo_raises`
- `test_copy_files_folder_requires_destination_suffix`

### Added in `tests/test_buckets_cli.py`

- `test_cp_remote_bucket_to_bucket`
- `test_cp_remote_repo_to_bucket`
- `test_cp_error_bucket_to_repo`
- `test_cp_error_remote_folder_requires_destination_suffix`

## Documentation

Updated:

- `docs/source/en/guides/buckets.md`
- `docs/source/en/guides/cli.md`
- `docs/source/en/package_reference/cli.md` (generated)

## Validation

Executed:

- `make style`
- `make quality`
- `pytest tests/test_buckets.py -k copy_files -q`
- `pytest tests/test_buckets_cli.py -k "cp_remote_bucket_to_bucket or cp_remote_repo_to_bucket or cp_error_bucket_to_repo or cp_error_remote_folder_requires_destination_suffix" -q`

All passed.
