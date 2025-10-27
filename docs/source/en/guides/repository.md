<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Create and manage a repository

The Hugging Face Hub is a collection of git repositories. [Git](https://git-scm.com/) is a widely used tool in software
development to easily version projects when working collaboratively. This guide will show you how to interact with the
repositories on the Hub, especially:

- Create and delete a repository.
- Manage branches and tags.
- Rename your repository.
- Update your repository visibility.
- Manage a local copy of your repository.

> [!WARNING]
> If you are used to working with platforms such as GitLab/GitHub/Bitbucket, your first instinct
> might be to use `git` CLI to clone your repo (`git clone`), commit changes (`git add, git commit`) and push them
> (`git push`). This is valid when using the Hugging Face Hub. However, software engineering and machine learning do
> not share the same requirements and workflows. Model repositories might maintain large model weight files for different
> frameworks and tools, so cloning the repository can lead to you maintaining large local folders with massive sizes. As
> a result, it may be more efficient to use our custom HTTP methods. You can read our [Git vs HTTP paradigm](../concepts/git_vs_http)
> explanation page for more details.

If you want to create and manage a repository on the Hub, your machine must be logged in. If you are not, please refer to
[this section](../quick-start#authentication). In the rest of this guide, we will assume that your machine is logged in.

## Repo creation and deletion

The first step is to know how to create and delete repositories. You can only manage repositories that you own (under
your username namespace) or from organizations in which you have write permissions.

### Create a repository

Create an empty repository with [`create_repo`] and give it a name with the `repo_id` parameter. The `repo_id` is your namespace followed by the repository name: `username_or_org/repo_name`.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

Or via CLI:

```bash
>>> hf repo create lysandre/test-model
Successfully created lysandre/test-model on the Hub.
Your repo is now available at https://huggingface.co/lysandre/test-model
```

By default, [`create_repo`] creates a model repository. But you can use the `repo_type` parameter to specify another repository type. For example, if you want to create a dataset repository:

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

Or via CLI:

```bash
>>> hf repo create lysandre/test-dataset --repo-type dataset
```

When you create a repository, you can set your repository visibility with the `private` parameter.

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", private=True)
```

Or via CLI:

```bash
>>> hf repo create lysandre/test-private --private
```

If you want to change the repository visibility at a later time, you can use the [`update_repo_settings`] function.

> [!TIP]
> If you are part of an organization with an Enterprise plan, you can create a repo in a specific resource group by passing `resource_group_id` as parameter to [`create_repo`]. Resource groups are a security feature to control which members from your org can access a given resource. You can get the resource group ID by copying it from your org settings page url on the Hub (e.g. `"https://huggingface.co/organizations/huggingface/settings/resource-groups/66670e5163145ca562cb1988"` => `"66670e5163145ca562cb1988"`). For more details about resource group, check out this [guide](https://huggingface.co/docs/hub/en/security-resource-groups).

### Delete a repository

Delete a repository with [`delete_repo`]. Make sure you want to delete a repository because this is an irreversible process!

Specify the `repo_id` of the repository you want to delete:

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

Or via CLI:

```bash
>>> hf repo delete lysandre/my-corrupted-dataset --repo-type dataset
```

### Duplicate a repository (only for Spaces)

In some cases, you want to copy someone else's repo to adapt it to your use case.
This is possible for Spaces using the [`duplicate_space`] method. It will duplicate the whole repository.
You will still need to configure your own settings (hardware, sleep-time, storage, variables and secrets). Check out our [Manage your Space](./manage-spaces) guide for more details.

```py
>>> from huggingface_hub import duplicate_space
>>> duplicate_space("multimodalart/dreambooth-training", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

## Upload and download files

Now that you have created your repository, you are interested in pushing changes to it and downloading files from it.

These 2 topics deserve their own guides. Please refer to the [upload](./upload) and the [download](./download) guides
to learn how to use your repository.


## Branches and tags

Git repositories often make use of branches to store different versions of a same repository.
Tags can also be used to flag a specific state of your repository, for example, when releasing a version.
More generally, branches and tags are referred as [git references](https://git-scm.com/book/en/v2/Git-Internals-Git-References).

### Create branches and tags

You can create new branch and tags using [`create_branch`] and [`create_tag`]:

```py
>>> from huggingface_hub import create_branch, create_tag

# Create a branch on a Space repo from `main` branch
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# Create a tag on a Dataset repo from `v0.1-release` branch
>>> create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

Or via CLI:

```bash
>>> hf repo branch create Matthijs/speecht5-tts-demo handle-dog-speaker --repo-type space
>>> hf repo tag create bigcode/the-stack v0.1.1 --repo-type dataset --revision v0.1-release -m "Bump release version."
```

You can use the [`delete_branch`] and [`delete_tag`] functions in the same way to delete a branch or a tag, or `hf repo branch delete` and `hf repo tag delete` respectively in CLI.


### List all branches and tags

You can also list the existing git refs from a repository using [`list_repo_refs`]:

```py
>>> from huggingface_hub import list_repo_refs
>>> list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## Change repository settings

Repositories come with some settings that you can configure. Most of the time, you will want to do that manually in the
repo settings page in your browser. You must have write access to a repo to configure it (either own it or being part of
an organization). In this section, we will see the settings that you can also configure programmatically using `huggingface_hub`.

Some settings are specific to Spaces (hardware, environment variables,...). To configure those, please refer to our [Manage your Spaces](../guides/manage-spaces) guide.

### Update visibility

A repository can be public or private. A private repository is only visible to you or members of the organization in which the repository is located. Change a repository to private as shown in the following:

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

Or via CLI:

```bash
>>> hf repo settings lysandre/test-private --private true
```

### Setup gated access

To give more control over how repos are used, the Hub allows repo authors to enable **access requests** for their repos. User must agree to share their contact information (username and email address) with the repo authors to access the files when enabled. A repo with access requests enabled is called a **gated repo**.

You can set a repo as gated using [`update_repo_settings`]:

```py
>>> from huggingface_hub import HfApi

>>> api = HfApi()
>>> api.update_repo_settings(repo_id=repo_id, gated="auto")  # Set automatic gating for a model
```

Or via CLI:

```bash
>>> hf repo settings lysandre/test-private --gated auto
```

### Rename your repository

You can rename your repository on the Hub using [`move_repo`]. Using this method, you can also move the repo from a user to
an organization. When doing so, there are a [few limitations](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo)
that you should be aware of. For example, you can't transfer your repo to another user.

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

Or via CLI:

```bash
>>> hf repo move Wauplin/cool-model huggingface/cool-model
```
