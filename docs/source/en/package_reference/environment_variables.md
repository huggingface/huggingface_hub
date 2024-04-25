<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Environment variables

`huggingface_hub` can be configured using environment variables.

If you are unfamiliar with environment variable, here are generic articles about them
[on macOS and Linux](https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/)
and on [Windows](https://phoenixnap.com/kb/windows-set-environment-variable).

This page will guide you through all environment variables specific to `huggingface_hub`
and their meaning.

## Generic

### HF_INFERENCE_ENDPOINT

To configure the inference api base url. You might want to set this variable if your organization
is pointing at an API Gateway rather than directly at the inference api.

Defaults to `"https://api-inference.huggingface.com"`.

### HF_HOME

To configure where `huggingface_hub` will locally store data. In particular, your token
and the cache will be stored in this folder.

Defaults to `"~/.cache/huggingface"` unless [XDG_CACHE_HOME](#xdgcachehome) is set.

### HF_HUB_CACHE

To configure where repositories from the Hub will be cached locally (models, datasets and
spaces).

Defaults to `"$HF_HOME/hub"` (e.g. `"~/.cache/huggingface/hub"` by default).

### HF_ASSETS_CACHE

To configure where [assets](../guides/manage-cache#caching-assets) created by downstream libraries
will be cached locally. Those assets can be preprocessed data, files downloaded from GitHub,
logs,...

Defaults to `"$HF_HOME/assets"` (e.g. `"~/.cache/huggingface/assets"` by default).

### HF_TOKEN

To configure the User Access Token to authenticate to the Hub. If set, this value will
overwrite the token stored on the machine (in either `$HF_TOKEN_PATH` or `"$HF_HOME/token"` if the former is not set).

For more details about authentication, check out [this section](../quick-start#authentication).

### HF_TOKEN_PATH

To configure where `huggingface_hub` should store the User Access Token. Defaults to `"$HF_HOME/token"` (e.g. `~/.cache/huggingface/token` by default).


### HF_HUB_VERBOSITY

Set the verbosity level of the `huggingface_hub`'s logger. Must be one of
`{"debug", "info", "warning", "error", "critical"}`.

Defaults to `"warning"`.

For more details, see [logging reference](../package_reference/utilities#huggingface_hub.utils.logging.get_verbosity).

### HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD

This environment variable has been deprecated and is now ignored by `huggingface_hub`. Downloading files to the local dir does not rely on symlinks anymore.

### HF_HUB_ETAG_TIMEOUT

Integer value to define the number of seconds to wait for server response when fetching the latest metadata from a repo before downloading a file. If the request times out, `huggingface_hub` will default to the locally cached files. Setting a lower value speeds up the workflow for machines with a slow connection that have already cached files. A higher value guarantees the metadata call to succeed in more cases. Default to 10s.

### HF_HUB_DOWNLOAD_TIMEOUT

Integer value to define the number of seconds to wait for server response when downloading a file. If the request times out, a TimeoutError is raised. Setting a higher value is beneficial on machine with a slow connection. A smaller value makes the process fail quicker in case of complete network outage. Default to 10s.

## Boolean values

The following environment variables expect a boolean value. The variable will be considered
as `True` if its value is one of `{"1", "ON", "YES", "TRUE"}` (case-insensitive). Any other value
(or undefined) will be considered as `False`.

### HF_HUB_OFFLINE

If set, no HTTP calls will me made to the Hugging Face Hub. If you try to download files, only the cached files will be accessed. If no cache file is detected, an error is raised This is useful in case your network is slow and you don't care about having the latest version of a file.

If `HF_HUB_OFFLINE=1` is set as environment variable and you call any method of [`HfApi`], an [`~huggingface_hub.utils.OfflineModeIsEnabled`] exception will be raised.

**Note:** even if the latest version of a file is cached, calling `hf_hub_download` still triggers a HTTP request to check that a new version is not available. Setting `HF_HUB_OFFLINE=1` will skip this call which speeds up your loading time.

### HF_HUB_DISABLE_IMPLICIT_TOKEN

Authentication is not mandatory for every requests to the Hub. For instance, requesting
details about `"gpt2"` model does not require to be authenticated. However, if a user is
[logged in](../package_reference/login), the default behavior will be to always send the token
in order to ease user experience (never get a HTTP 401 Unauthorized) when accessing private or gated repositories. For privacy, you can
disable this behavior by setting `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. In this case,
the token will be sent only for "write-access" calls (example: create a commit).

**Note:** disabling implicit sending of token can have weird side effects. For example,
if you want to list all models on the Hub, your private models will not be listed. You
would need to explicitly pass `token=True` argument in your script.

### HF_HUB_DISABLE_PROGRESS_BARS

For time consuming tasks, `huggingface_hub` displays a progress bar by default (using tqdm).
You can disable all the progress bars at once by setting `HF_HUB_DISABLE_PROGRESS_BARS=1`.

### HF_HUB_DISABLE_SYMLINKS_WARNING

If you are on a Windows machine, it is recommended to enable the developer mode or to run
`huggingface_hub` in admin mode. If not, `huggingface_hub` will not be able to create
symlinks in your cache system. You will be able to execute any script but your user experience
will be degraded as some huge files might end-up duplicated on your hard-drive. A warning
message is triggered to warn you about this behavior. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`,
to disable this warning.

For more details, see [cache limitations](../guides/manage-cache#limitations).

### HF_HUB_DISABLE_EXPERIMENTAL_WARNING

Some features of `huggingface_hub` are experimental. This means you can use them but we do not guarantee they will be
maintained in the future. In particular, we might update the API or behavior of such features without any deprecation
cycle. A warning message is triggered when using an experimental feature to warn you about it. If you're comfortable debugging any potential issues using an experimental feature, you can set `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` to disable the warning.

If you are using an experimental feature, please let us know! Your feedback can help us design and improve it.

### HF_HUB_DISABLE_TELEMETRY

By default, some data is collected by HF libraries (`transformers`, `datasets`, `gradio`,..) to monitor usage, debug issues and help prioritize features.
Each library defines its own policy (i.e. which usage to monitor) but the core implementation happens in `huggingface_hub` (see [`send_telemetry`]).

You can set `HF_HUB_DISABLE_TELEMETRY=1` as environment variable to globally disable telemetry.

### HF_HUB_ENABLE_HF_TRANSFER

Set to `True` for faster uploads and downloads from the Hub using `hf_transfer`.

By default, `huggingface_hub` uses the Python-based `requests.get` and `requests.post` functions.
Although these are reliable and versatile,
they may not be the most efficient choice for machines with high bandwidth.
[`hf_transfer`](https://github.com/huggingface/hf_transfer) is a Rust-based package developed to
maximize the bandwidth used by dividing large files into smaller parts
and transferring them simultaneously using multiple threads.
This approach can potentially double the transfer speed.
To use `hf_transfer`:

1. Specify the `hf_transfer` extra when installing `huggingface_hub`
   (e.g. `pip install huggingface_hub[hf_transfer]`).
2. Set `HF_HUB_ENABLE_HF_TRANSFER=1` as an environment variable.

Please note that using `hf_transfer` comes with certain limitations. Since it is not purely Python-based, debugging errors may be challenging. Additionally, `hf_transfer` lacks several user-friendly features such as resumable downloads and proxies. These omissions are intentional to maintain the simplicity and speed of the Rust logic. Consequently, `hf_transfer` is not enabled by default in `huggingface_hub`.

## Deprecated environment variables

In order to standardize all environment variables within the Hugging Face ecosystem, some variables have been marked as deprecated. Although they remain functional, they no longer take precedence over their replacements. The following table outlines the deprecated variables and their corresponding alternatives:


| Deprecated Variable | Replacement |
| --- | --- |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` |
| `HUGGINGFACE_ASSETS_CACHE` | `HF_ASSETS_CACHE` |
| `HUGGING_FACE_HUB_TOKEN` | `HF_TOKEN` |
| `HUGGINGFACE_HUB_VERBOSITY` | `HF_HUB_VERBOSITY` |

## From external tools

Some environment variables are not specific to `huggingface_hub` but are still taken into account when they are set.

### DO_NOT_TRACK

Boolean value. Equivalent to `HF_HUB_DISABLE_TELEMETRY`. When set to true, telemetry is globally disabled in the Hugging Face Python ecosystem (`transformers`, `diffusers`, `gradio`, etc.). See https://consoledonottrack.com/ for more details.

### NO_COLOR

Boolean value. When set, `huggingface-cli` tool will not print any ANSI color.
See [no-color.org](https://no-color.org/).

### XDG_CACHE_HOME

Used only when `HF_HOME` is not set!

This is the default way to configure where [user-specific non-essential (cached) data should be written](https://wiki.archlinux.org/title/XDG_Base_Directory)
on linux machines.

If `HF_HOME` is not set, the default home will be `"$XDG_CACHE_HOME/huggingface"` instead
of `"~/.cache/huggingface"`.
