---
title: How to download files from the Hub
---

# How to integrate downstream utilities in your library

Utilities that allow your library to download files from the Hub are referred to as *downstream* utilities. This guide introduces additional downstream utilities you can integrate with your library, or use separately on their own. You will learn how to:

* Retrieve a URL to download.
* Download a file and cache it on your disk.
* Download all the files in a repository.

## hf_hub_url

Use `hf_hub_url` to retrieve the URL of a specific file to download by providing a `filename`.

![/docs/assets/hub/repo.png](/docs/assets/hub/repo.png)

```python
>>> from huggingface_hub import hf_hub_url
>>> hf_hub_url(repo_id="lysandre/arxiv-nlp", filename="config.json")
'https://huggingface.co/lysandre/arxiv-nlp/resolve/main/config.json'
```

Specify a particular file version by providing the file revision. The file revision can be a branch, a tag, or a commit hash.

When using the commit hash, it must be the full-length hash instead of a 7-character commit hash:

```python
>>> hf_hub_url(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
'https://huggingface.co/lysandre/arxiv-nlp/resolve/877b84a8f93f2d619faa2a6e514a32beef88ab0a/config.json'
```

`hf_hub_url` can also use the branch name to specify a file revision:

```python
hf_hub_url(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="main")
```

Specify a file revision with a tag identifier. For example, if you want `v1.0` of the `config.json` file:

```python
hf_hub_url(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")
```

## cached_download

`cached_download` is useful for downloading and caching a file on your local disk. Once stored in your cache, you don't have to redownload the file the next time you use it. `cached_download` is a hands-free solution for staying up to date with new file versions. When a downloaded file is updated in the remote repository, `cached_download` will automatically download and store it for you.

Begin by retrieving your file URL with `hf_hub_url`, and then pass the specified URL to `cached_download` to download the file:

```python
>>> from huggingface_hub import hf_hub_url, cached_download
>>> config_file_url = hf_hub_url("lysandre/arxiv-nlp", filename="config.json")
>>> cached_download(config_file_url)
'/home/lysandre/.cache/huggingface/hub/bc0e8cc2f8271b322304e8bb84b3b7580701d53a335ab2d75da19c249e2eeebb.066dae6fdb1e2b8cce60c35cc0f78ed1451d9b341c78de19f3ad469d10a8cbb1'
```

`hf_hub_url` and `cached_download` work hand in hand to download a file. This is precisely how `hf_hub_download` from the tutorial works! `hf_hub_download` is simply a wrapper that calls both `hf_hub_url` and `cached_download`.

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
```

## snapshot_download

`snapshot_download` downloads an entire repository at a given revision. Like `cached_download`, all downloaded files are cached on your local disk. However, even if only a single file is updated, the entire repository will be redownloaded.

Download a whole repository as shown in the following:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/lysandre__arxiv-nlp.894a9adde21d9a3e3843e6d5aeaaf01875c7fade'
```

`snapshot_download` downloads the latest revision by default. If you want a specific repository revision, use the `revision` parameter as shown with `hf_hub_url`.

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="main")
```

In general, it is usually better to manually download files with `hf_hub_download` (if you already know the file name) to avoid re-downloading an entire repository. `snapshot_download` is helpful when your library's downloading utility is a helper, and unaware of which files need to be downloaded.

However, you don't want to always download the contents of an entire repository with `snapshot_download`. Even if you don't know the file name and only know the file type, you can download specific files with `allow_regex` and `ignore_regex`.
use of the `allow_regex` and `ignore_regex` arguments to specify 
which files shall be downloaded.
`allow_regex` and `ignore_regex` accept either a single regex or a list of regexes. 
The regex matching is based on [`fnmatch`](https://docs.python.org/3/library/fnmatch.html) which means it provides support for Unix shell-style wildcards.

For example, you can use `allow_regex` to only download JSON configuration files:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_regex="*.json")
```

On the other hand, `ignore_regex` can be used to exclude certain files from being downloaded. The following example ignores the `.msgpack` and `.h5` file extensions:
or `.h5` extensions, you could make use of `ignore_regex`:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_regex=["*.msgpack", "*.h5"])
```

Passing a regex can be especially useful when repositories contain files that 
are never expected to be downloaded by `snapshot_download`.

Note that passing `allow_regex` or `ignore_regex` does **not** prevent 
`snapshot_download` from re-downloading the entire model repository if an ignored
file is changed.
