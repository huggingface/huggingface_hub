## Summary

This PR adds a new `HfApi.copy_files` API and extends `hf buckets cp` to support remote HF-handle copy workflows.

### New capability

- Copy from bucket to bucket (same bucket or different bucket)
- Copy from repo (model/dataset/space) to bucket
- Reject bucket->repo and repo->repo destinations (not supported yet)

## API changes

### Added

- `HfApi.copy_files(source, destination, *, token=None) -> None`
- top-level alias export: `copy_files`

### Handle support

- Source and destination accept HF handles (`hf://...`)
- Repo source handles support explicit repo type prefixes and optional `@revision`

Examples:

- `hf://buckets/<namespace>/<bucket_id>/path/to/file`
- `hf://datasets/<namespace>/<dataset_id>/path/to/folder/`
- `hf://<namespace>/<model_id>/path/to/file`
- `hf://<namespace>/<model_id>@<revision>/path/to/file`

## Copy behavior

- File source: copy one file
- Folder source: recursively copy files under the source folder
- Folder copy requires destination to end with `/`

### Content transfer strategy

- Repo source file with `xet_hash`: copied directly by hash
- Repo source file without `xet_hash` (regular small file): download then re-upload
- Bucket source to same bucket: copied by hash
- Bucket source to different bucket: download then re-upload fallback

## Internal update

Extended `_batch_bucket_files` internals to accept prebuilt `_BucketAddFile` entries, allowing direct hash-based add operations when hash metadata is already known.

## CLI changes

Updated `hf buckets cp`:

- now supports remote->remote HF handle copies via `api.copy_files`
- preserves existing local<->bucket and stdin/stdout behavior
- remote copy output (non-quiet):
  - `Copied: SRC -> DST`

## Tests added

### `tests/test_buckets.py`

- `test_copy_files_bucket_to_same_bucket_file`
- `test_copy_files_bucket_to_different_bucket_folder`
- `test_copy_files_repo_to_bucket_with_revision`
- `test_copy_files_bucket_to_repo_raises`
- `test_copy_files_folder_requires_destination_suffix`

### `tests/test_buckets_cli.py`

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

Executed successfully:

- `make style`
- `make quality`
- `pytest tests/test_buckets.py -k copy_files -q`
- `pytest tests/test_buckets_cli.py -k "cp_remote_bucket_to_bucket or cp_remote_repo_to_bucket or cp_error_bucket_to_repo or cp_error_remote_folder_requires_destination_suffix" -q`
