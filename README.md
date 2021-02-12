## `huggingface_hub`

### Client library to download and publish models and other files on the huggingface.co hub

<p align="center">
	<img alt="Build" src="https://github.com/huggingface/huggingface_hub/workflows/Python%20tests/badge.svg">
	<a href="https://github.com/huggingface/huggingface_hub/blob/master/LICENSE">
		<img alt="GitHub" src="https://img.shields.io/github/license/huggingface/huggingface_hub.svg?color=blue">
	</a>
	<a href="https://github.com/huggingface/huggingface_hub/releases">
		<img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg">
	</a>
</p>

> **Do you have an open source ML library?**
> We're looking to partner with a small number of other cool open source ML libraries to provide model hosting + versioning. 
> https://twitter.com/julien_c/status/1336374565157679104 https://twitter.com/mnlpariente/status/1336277058062852096
>
> Advantages are:
> - versioning is built-in (as hosting is built around git and git-lfs), no lock-in, you can just `git clone` away.
> - anyone can upload a new model for your library, just need to add the corresponding tag for the model to be discoverable – no more need for a hardcoded list in your code
> - Fast downloads! We use Cloudfront (a CDN) to geo-replicate downloads so they're blazing fast from anywhere on the globe
> - Usage stats and more features to come
>
> Ping us if interested 😎

<br>

### ♻️ Partial list of implementations in third party libraries:

- http://github.com/asteroid-team/asteroid [[initial PR 👀](https://github.com/asteroid-team/asteroid/pull/377)]
- https://github.com/pyannote/pyannote-audio [[initial PR 👀](https://github.com/pyannote/pyannote-audio/pull/549)]
- https://github.com/flairNLP/flair [[work-in-progress, initial PR 👀](https://github.com/flairNLP/flair/pull/1974)]
- https://github.com/espnet/espnet [[initial PR 👀](https://github.com/espnet/espnet/pull/2815)]

<br>

## Download files from the huggingface.co hub

Integration inside a library is super simple. We expose two functions, `hf_hub_url()` and `cached_download()`.

### `hf_hub_url`

`hf_hub_url()` takes:
- a repo id (e.g. a model id like `julien-c/EsperBERTo-small` i.e. a user or organization name and a repo name, separated by `/`),
- a filename (like `pytorch_model.bin`),
- and an optional git revision id (can be a branch name, a tag, or a commit hash)

and returns the url we'll use to download the actual files: `https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin`

If you check out this URL's headers with a `HEAD` http request (which you can do from the command line with `curl -I`) for a few different files, you'll see that:
- small files are returned directly
- large files (i.e. the ones stored through [git-lfs](https://git-lfs.github.com/)) are returned via a redirect to a Cloudfront URL. Cloudfront is a Content Delivery Network, or CDN, that ensures that downloads are as fast as possible from anywhere on the globe.

### `cached_download`

`cached_download()` takes the following parameters, downloads the remote file, stores it to disk (in a versioning-aware way) and returns its local file path.

Parameters:
- a remote `url`
- your library's name and version (`library_name` and `library_version`), which will be added to the HTTP requests' user-agent so that we can provide some usage stats.
- a `cache_dir` which you can specify if you want to control where on disk the files are cached.

Check out the source code for all possible params (we'll create a real doc page in the future).

<br>

## Publish models to the huggingface.co hub

Uploading a model to the hub is super simple too:
- create a model repo directly from the website, at huggingface.co/new (models can be public or private, and are namespaced under either a user or an organization)
- clone it with git
- [download and install git lfs](https://git-lfs.github.com/) if you don't already have it on your machine (you can check by running a simple `git lfs`)
- add, commit and push your files, from git, as you usually do.

**We are intentionally not wrapping git too much, so that you can go on with the workflow you’re used to and the tools you already know.**

> 👀 To see an example of how we document the model sharing process in `transformers`, check out https://huggingface.co/transformers/model_sharing.html

Users add tags into their README.md model cards (e.g. your `library_name`, a domain tag like `audio`, etc.) to make sure their models are discoverable.

**Documentation about the model hub itself is at https://huggingface.co/docs**

### API utilities in `hf_api.py`

You don't need them for the standard publishing workflow, however, if you need a programmatic way of creating a repo, deleting it (`⚠️ caution`), or listing models from the hub, you'll find helpers in `hf_api.py`.

We also have an API to query models by specific tags (e.g. if you want to list models compatible to your library)

### `huggingface-cli`

Those API utilities are also exposed through a CLI:

```bash
huggingface-cli login
huggingface-cli logout
huggingface-cli whoami
huggingface-cli repo create
```

### Need to upload large (>5GB) files?

To upload large files (>5GB 🔥), you need to install the custom transfer agent for git-lfs, bundled in this package. 

To install, just run:

```bash
$ huggingface-cli lfs-enable-largefiles
```

This should be executed once for each model repo that contains a model file >5GB. If you just try to push a file bigger than 5GB without running that command, you will get an error with a message reminding you to run it.

Finally, there's a `huggingface-cli lfs-multipart-upload` command but that one is internal (called by lfs directly) and is not meant to be called by the user.

<br>

## Visual integration into the huggingface.co hub

Finally, we'll implement a few tweaks to improve the UX for your models on the website – let's use [Asteroid](https://github.com/asteroid-team/asteroid) as an example:

![asteroid-model](https://cdn-media.huggingface.co/huggingface_hub/asteroid-model-optim.png)

Model authors add an `asteroid` tag to their model card and they get the advantages of model versioning built-in

![use-in-asteroid](https://cdn-media.huggingface.co/huggingface_hub/use-in-asteroid.png)

We add a custom "Use in Asteroid" button.

![asteroid-code-sample](https://cdn-media.huggingface.co/huggingface_hub/asteroid-code-sample.png)

When clicked you get a library-specific code sample that you'll be able to specify. 🔥

<br>

## Feedback (feature requests, bugs, etc.) is super welcome 💙💚💛💜♥️🧡
