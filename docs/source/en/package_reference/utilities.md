<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Utilities

## Configure logging

The `huggingface_hub` package exposes a `logging` utility to control the logging level of the package itself.
You can import it as such:

```py
from huggingface_hub import logging
```

Then, you may define the verbosity in order to update the amount of logs you'll see:

```python
from huggingface_hub import logging

logging.set_verbosity_error()
logging.set_verbosity_warning()
logging.set_verbosity_info()
logging.set_verbosity_debug()

logging.set_verbosity(...)
```

The levels should be understood as follows:

- `error`: only show critical logs about usage which may result in an error or unexpected behavior.
- `warning`: show logs that aren't critical but usage may result in unintended behavior.
  Additionally, important informative logs may be shown.
- `info`: show most logs, including some verbose logging regarding what is happening under the hood.
  If something is behaving in an unexpected manner, we recommend switching the verbosity level to this in order
  to get more information.
- `debug`: show all logs, including some internal logs which may be used to track exactly what's happening
  under the hood.

[[autodoc]] logging.get_verbosity
[[autodoc]] logging.set_verbosity
[[autodoc]] logging.set_verbosity_info
[[autodoc]] logging.set_verbosity_debug
[[autodoc]] logging.set_verbosity_warning
[[autodoc]] logging.set_verbosity_error
[[autodoc]] logging.disable_propagation
[[autodoc]] logging.enable_propagation

### Repo-specific helper methods

The methods exposed below are relevant when modifying modules from the `huggingface_hub` library itself.
Using these shouldn't be necessary if you use `huggingface_hub` and you don't modify them.

[[autodoc]] logging.get_logger

## Configure progress bars

Progress bars are a useful tool to display information to the user while a long-running task is being executed (e.g.
when downloading or uploading files). `huggingface_hub` exposes a [`~utils.tqdm`] wrapper to display progress bars in a
consistent way across the library.

By default, progress bars are enabled. You can disable them globally by setting `HF_HUB_DISABLE_PROGRESS_BARS`
environment variable. You can also enable/disable them using [`~utils.enable_progress_bars`] and
[`~utils.disable_progress_bars`]. If set, the environment variable has priority on the helpers.

```py
>>> from huggingface_hub import snapshot_download
>>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

>>> # Disable progress bars globally
>>> disable_progress_bars()

>>> # Progress bar will not be shown !
>>> snapshot_download("gpt2")

>>> are_progress_bars_disabled()
True

>>> # Re-enable progress bars globally
>>> enable_progress_bars()
```

### Group-specific control of progress bars

You can also enable or disable progress bars for specific groups. This allows you to manage progress bar visibility more granularly within different parts of your application or library. When a progress bar is disabled for a group, all subgroups under it are also affected unless explicitly overridden.

```py
# Disable progress bars for a specific group
>>> disable_progress_bars("peft.foo")
>>> assert not are_progress_bars_disabled("peft")
>>> assert not are_progress_bars_disabled("peft.something")
>>> assert are_progress_bars_disabled("peft.foo")
>>> assert are_progress_bars_disabled("peft.foo.bar")

# Re-enable progress bars for a subgroup
>>> enable_progress_bars("peft.foo.bar")
>>> assert are_progress_bars_disabled("peft.foo")
>>> assert not are_progress_bars_disabled("peft.foo.bar")

# Use groups with tqdm
# No progress bar for `name="peft.foo"`
>>> for _ in tqdm(range(5), name="peft.foo"):
...     pass

# Progress bar will be shown for `name="peft.foo.bar"`
>>> for _ in tqdm(range(5), name="peft.foo.bar"):
...     pass
100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]
```

### are_progress_bars_disabled

[[autodoc]] huggingface_hub.utils.are_progress_bars_disabled

### disable_progress_bars

[[autodoc]] huggingface_hub.utils.disable_progress_bars

### enable_progress_bars

[[autodoc]] huggingface_hub.utils.enable_progress_bars

## Configure HTTP backend

In some environments, you might want to configure how HTTP calls are made, for example if you are using a proxy.
`huggingface_hub` let you configure this globally using [`configure_http_backend`]. All requests made to the Hub will
then use your settings. Under the hood, `huggingface_hub` uses `requests.Session` so you might want to refer to the
[`requests` documentation](https://requests.readthedocs.io/en/latest/user/advanced) to learn more about the available
parameters.

Since `requests.Session` is not guaranteed to be thread-safe, `huggingface_hub` creates one session instance per thread.
Using sessions allows us to keep the connection open between HTTP calls and ultimately save time. If you are
integrating `huggingface_hub` in a third-party library and wants to make a custom call to the Hub, use [`get_session`]
to get a Session configured by your users (i.e. replace any `requests.get(...)` call by `get_session().get(...)`).

[[autodoc]] configure_http_backend

[[autodoc]] get_session


## Handle HTTP errors

`huggingface_hub` defines its own HTTP errors to refine the `HTTPError` raised by
`requests` with additional information sent back by the server.

### Raise for status

[`~utils.hf_raise_for_status`] is meant to be the central method to "raise for status" from any
request made to the Hub. It wraps the base `requests.raise_for_status` to provide
additional information. Any `HTTPError` thrown is converted into a `HfHubHTTPError`.

```py
import requests
from huggingface_hub.utils import hf_raise_for_status, HfHubHTTPError

response = requests.post(...)
try:
    hf_raise_for_status(response)
except HfHubHTTPError as e:
    print(str(e)) # formatted message
    e.request_id, e.server_message # details returned by server

    # Complete the error message with additional information once it's raised
    e.append_to_message("\n`create_commit` expects the repository to exist.")
    raise
```

[[autodoc]] huggingface_hub.utils.hf_raise_for_status

### HTTP errors

Here is a list of HTTP errors thrown in `huggingface_hub`.

#### HfHubHTTPError

`HfHubHTTPError` is the parent class for any HF Hub HTTP error. It takes care of parsing
the server response and format the error message to provide as much information to the
user as possible.

[[autodoc]] huggingface_hub.utils.HfHubHTTPError

#### RepositoryNotFoundError

[[autodoc]] huggingface_hub.utils.RepositoryNotFoundError

#### GatedRepoError

[[autodoc]] huggingface_hub.utils.GatedRepoError

#### RevisionNotFoundError

[[autodoc]] huggingface_hub.utils.RevisionNotFoundError

#### EntryNotFoundError

[[autodoc]] huggingface_hub.utils.EntryNotFoundError

#### BadRequestError

[[autodoc]] huggingface_hub.utils.BadRequestError

#### LocalEntryNotFoundError

[[autodoc]] huggingface_hub.utils.LocalEntryNotFoundError

#### OfflineModeIsEnabled

[[autodoc]] huggingface_hub.utils.OfflineModeIsEnabled

## Telemetry

`huggingface_hub` includes an helper to send telemetry data. This information helps us debug issues and prioritize new features.
Users can disable telemetry collection at any time by setting the `HF_HUB_DISABLE_TELEMETRY=1` environment variable.
Telemetry is also disabled in offline mode (i.e. when setting HF_HUB_OFFLINE=1).

If you are maintainer of a third-party library, sending telemetry data is as simple as making a call to [`send_telemetry`].
Data is sent in a separate thread to reduce as much as possible the impact for users.

[[autodoc]] utils.send_telemetry


## Validators

`huggingface_hub` includes custom validators to validate method arguments automatically.
Validation is inspired by the work done in [Pydantic](https://pydantic-docs.helpmanual.io/)
to validate type hints but with more limited features.

### Generic decorator

[`~utils.validate_hf_hub_args`] is a generic decorator to encapsulate
methods that have arguments following `huggingface_hub`'s naming. By default, all
arguments that has a validator implemented will be validated.

If an input is not valid, a [`~utils.HFValidationError`] is thrown. Only
the first non-valid value throws an error and stops the validation process.

Usage:

```py
>>> from huggingface_hub.utils import validate_hf_hub_args

>>> @validate_hf_hub_args
... def my_cool_method(repo_id: str):
...     print(repo_id)

>>> my_cool_method(repo_id="valid_repo_id")
valid_repo_id

>>> my_cool_method("other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> my_cool_method(repo_id="other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> @validate_hf_hub_args
... def my_cool_auth_method(token: str):
...     print(token)

>>> my_cool_auth_method(token="a token")
"a token"

>>> my_cool_auth_method(use_auth_token="a use_auth_token")
"a use_auth_token"

>>> my_cool_auth_method(token="a token", use_auth_token="a use_auth_token")
UserWarning: Both `token` and `use_auth_token` are passed (...). `use_auth_token` value will be ignored.
"a token"
```

#### validate_hf_hub_args

[[autodoc]] utils.validate_hf_hub_args

#### HFValidationError

[[autodoc]] utils.HFValidationError

### Argument validators

Validators can also be used individually. Here is a list of all arguments that can be
validated.

#### repo_id

[[autodoc]] utils.validate_repo_id

#### smoothly_deprecate_use_auth_token

Not exactly a validator, but ran as well.

[[autodoc]] utils.smoothly_deprecate_use_auth_token
