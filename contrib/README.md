# Contrib test suite

The contrib folder contains simple end-to-end scripts to test integration of `huggingface_hub` in downstream libraries. The main goal is to proactively notice breaking changes and deprecation warnings.

## Add tests for a new library

To add another contrib lib, one must:
1. Create a subfolder with the lib name. Example: `./contrib/transformers`
2. Create a `requirements.txt` file specific to this lib. Example `./contrib/transformers/requirements.txt`
3. Implements tests for this lib. Example: `./contrib/transformers/test_push_to_hub.py`
4. Run `make style`. This will edit both `makefile` and `.github/workflows/contrib-tests.yml` to add the lib to list of libs to test. Make sure changes are accurate before committing.

## Run contrib tests on CI

Contrib tests can be [manually triggered in GitHub](https://github.com/huggingface/huggingface_hub/actions) with the `Contrib tests` workflow.

Tests are not run in the default test suite (for each PR) as this would slow down development process. The goal is to notice breaking changes, not to avoid them. In particular, it is interesting to trigger it before a release to make sure it will not cause too much friction.

## Run contrib tests locally

Tests must be ran individually for each dependent library. Here is an example to run
`timm` tests. Tests are separated to avoid conflicts between version dependencies.

### Run all contrib tests

Before running tests, a virtual env must be setup for each contrib library. To do so, run:

```sh
# Run setup in parallel to save time
make contrib_setup -j4
```

Then tests can be run

```sh
# Optional: -j4 to run in parallel. Output will be messy in that case.
make contrib_test -j4
```

Optionally, it is possible to setup and run all tests in a single command. However this
take more time as you don't need to setup the venv each time you run tests.

```sh
make contrib -j4
```

Finally, it is possible to delete all virtual envs to get a fresh start for contrib tests.
After running this command, `contrib_setup` will have to re-download/re-install all dependencies.

```
make contrib_clear
```

### Run contrib tests for a single lib

Instead of running tests for all contrib libraries, you can run a specific lib:

```sh
# Setup timm tests
make contrib_setup_timm

# Run timm tests
make contrib_test_timm

# (or) Setup and run timm tests at once
make contrib_timm

# Delete timm virtualenv if corrupted
make contrib_clear_timm
```
