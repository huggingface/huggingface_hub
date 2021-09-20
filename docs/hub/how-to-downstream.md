# How to integrate downstream methods in your library

In the tutorial, you learned some basic functions for integrating the Hub into your library. Functions that allow your library to download files from the Hub are referred to as *downstream* functions. This guide introduces additional downstream methods you can integrate in your library. You will learn how to:

* Download a file without caching it on your disk.
* Download a file and cache it on your disk.
* Download all the files in a repository.

## `hf_hub_url`

Use `hf_hub_url` to download a specific file from a repository by providing a `filename`. This function takes the URL of the file, and downloads it. For example, if you want to download a `config.json` file from a repository:

![/docs/assets/hub/repo.png](/docs/assets/hub/repo.png)

```python
>>> from huggingface_hub import hf_hub_url
>>> hf_hub_url(repo_id="lysandre/arxiv-nlp", filename="config.json")
'https://huggingface.co/lysandre/arxiv-nlp/resolve/main/config.json'
```

Download a specific file version by providing the file revision. The file revision can be a branch, a tag, or a commit hash. The following example shows how to download a specific version of the `config.json` file using the commit hash:

```python
>>> hf_hub_url(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
'https://huggingface.co/lysandre/arxiv-nlp/resolve/877b84a8f93f2d619faa2a6e514a32beef88ab0a/config.json'
```

## `cached_download`

The `cached_download` function is useful for caching a file on your local disk. Once it is stored in your cache, you don't have to redownload the file the next time you use it. This is a hands-free solution for staying up to date with new file versions. When one of your downloaded files is updated in the repository, it is automatically downloaded and stored for you.

Begin by downloading your file with `hf_hub_url`, and then pass the specified URL to `cached_download`:

```python
>>> from huggingface_hub import hf_hub_url, cached_download
>>> config_file_url = hf_hub_url("lysandre/arxiv-nlp", filename="config.json")
>>> cached_download(config_file_url)
'/home/lysandre/.cache/huggingface/hub/bc0e8cc2f8271b322304e8bb84b3b7580701d53a335ab2d75da19c249e2eeebb.066dae6fdb1e2b8cce60c35cc0f78ed1451d9b341c78de19f3ad469d10a8cbb1'
```

## `snapshot_download`

The `snapshot_download` function works well for downloading an entire repository. Like the previous functions, all downloaded files are cached on your local disk, and will be automatically updated if a file in the repository is changed.

Download a whole repository as shown in the following:

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/lysandre__arxiv-nlp.894a9adde21d9a3e3843e6d5aeaaf01875c7fade'
```