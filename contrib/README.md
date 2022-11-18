# Contrib test suite

The contrib folder contains simple end-to-end scripts to test integration of `huggingface_hub` in downstream libraries. The main goal is to proactively notice breaking changes and deprecation warnings.

## Run contrib tests on CI

Contrib tests can be [manually triggered in github](https://github.com/huggingface/huggingface_hub/actions) with the `Contrib tests` workflow.

Tests are not run in the default test suite (for each PR) as this would slow down development process. The goal is to notice breaking changes, not to avoid them. In particular, it is interesting to trigger it before a release to make sure it will not cause too much friction.

## Run contrib tests locally

### Install dependencies

```sh
# Create a separate contrib environment
python3 -m venv .venv_contrib
source .venv_contrib/bin/activate

# Install requirements
pip install . # huggingface_hub
pip install -r contrib/requirements.txt

# Run tests !
pytest contrib -n 4
```