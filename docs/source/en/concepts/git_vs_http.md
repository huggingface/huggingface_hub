<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Git vs HTTP paradigm

The `huggingface_hub` library is a library for interacting with the Hugging Face Hub, which is a collection of git-based repositories (models, datasets or Spaces). There are two main ways to access the Hub using `huggingface_hub`.

The first approach, the so-called "git-based" approach, relies on using standard `git` commands directly in a terminal. This method allows you to clone repositories, create commits, and push changes manually. The second option, called the "HTTP-based" approach, involves making HTTP requests using the [`HfApi`] client. Let's examine the pros and cons of each approach.

## Git: the historical CLI-based approach

At first, most users interacted with the Hugging Face Hub using plain `git` commands such as `git clone`, `git add`, `git commit`, `git push`, `git tag`, or `git checkout`.

This approach lets you work with a full local copy of the repository on your machine, just like in traditional software development. This can be an advantage when you need offline access or want to work with the full history of a repository. However, it also comes with downsides: you are responsible for keeping the repository up-to-date locally, handling credentials, and managing large files (via `git-lfs`), which can become cumbersome when working with large machine learning models or datasets.

In many machine learning workflows, you may only need to download a few files for inference or convert weights without needing to clone the entire repository. In such cases, using `git` can be overkill and introduce unnecessary complexity.

## HfApi: a flexible and convenient HTTP client

The [`HfApi`] class was developed to provide an alternative to using local git repositories, which can be cumbersome to maintain, especially when dealing with large models or datasets. The [`HfApi`] class offers the same functionality as git-based workflows -such as downloading and pushing files and creating branches and tags- but without the need for a local folder that needs to be kept in sync.

In addition to the functionalities already provided by `git`, the [`HfApi`] class offers additional features, such as the ability to manage repos, download files using caching for efficient reuse, search the Hub for repos and metadata, access community features such as discussions, PRs, and comments, and configure Spaces hardware and secrets.

## What should I use ? And when ?

Overall, the **HTTP-based approach is the recommended way to use** `huggingface_hub` in all cases. [`HfApi`] allows you to pull and push changes, work with PRs, tags and branches, interact with discussions and much more.

However, not all git commands are available through [`HfApi`]. Some may never be implemented, but we are always trying to improve and close the gap. If you don't see your use case covered, please open [an issue on GitHub](https://github.com/huggingface/huggingface_hub)! We welcome feedback to help build the HF ecosystem with and for our users.

This preference for the HTTP-based [`HfApi`] over direct `git` commands does not mean that git versioning will disappear from the Hugging Face Hub anytime soon. It will always be possible to use `git` locally in workflows where it makes sense.