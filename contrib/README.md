# Contrib test suite

The contrib folder contains simple end-to-end scripts to test integration of `huggingface_hub` in downstream libraries. The main goal is to proactively notice breaking changes and deprecation warnings.

## Run contrib tests on CI

Contrib tests can be [manually triggered in github](https://github.com/huggingface/huggingface_hub/actions) with the `Contrib tests` workflow.

Tests are not run in the default test suite (for each PR) as this would slow down development process. The goal is to notice breaking changes, not to avoid them. In particular, it is interesting to trigger it before a release to make sure it will not cause too much friction.

## Run contrib tests locally

Tests must be ran individually for each dependent library. Here is an example to run
`timm` tests. Tests are separated to avoid conflicts between version dependencies.

### Install dependencies

```sh
# Create a separate contrib environment
python3 -m venv .venv_contrib_timm
source .venv_contrib_timm/bin/activate

# Install requirements
pip install -r contrib/reqs_common.txt
pip install -r contrib/reqs_timm.txt
pip install . # huggingface_hub
```

### Run tests !

```
pytest contrib/test_timm.py
```