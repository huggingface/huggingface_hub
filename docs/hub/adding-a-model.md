---
title: Adding your model to the Hugging Face Hub
---

# Adding your model to the Hugging Face Hub

## Why?

Uploading models to the Hugging Face Hub has many [benefits](https://huggingface.co/docs/hub/main#whats-a-repository). In summary, you get versioning, branches, discoverability and sharing features, integration with over a dozen libraries and more!

## Accounts and organizations

The first step is to create an account at [Hugging Face](https://huggingface.co/login). The models are shared in the form of git-based repositories. You have control over your repository, so you can have checkpoints, configs and any files you might want to upload.

The repository can be either linked with an individual, such as [osanseviero/fashion_brands_patterns](https://huggingface.co/osanseviero/fashion_brands_patterns) or with an organization, such as [facebook/bart-large-xsum](https://huggingface.co/facebook/bart-large-xsum). Organizations can be used if you want to upload models that are related to a company, community or library! If you choose an organization, the model will be featured on the organization’s page and every member of the organization will have the ability to contribute to the repository. You can create a new organization [here](https://huggingface.co/organizations/new).

## Creating a repository

Using the web interface, you can easily create repositories, add files (even large ones!), explore models, visualize diffs, and much more. Let's begin by creating a repository.

1. To create a new repository, visit [huggingface.co/new](http://huggingface.co/new):

![/docs/assets/hub/new_repo.png](/docs/assets/hub/new_repo.png)

2. First, specify the owner of the repository: this can be either you or any of the organizations you’re affiliated with. 

3. Next, enter your model’s name. This will also be the name of the repository. Finally, you can specify whether you want your model to be public or private.

After creating your model repository, you should see a page like this:

![/docs/assets/hub/empty_repo.png](/docs/assets/hub/empty_repo.png)

4. This is where your model will be hosted. To start populating it, you can add a README file directly from the web interface.

![/docs/assets/hub/repo_readme.png](/docs/assets/hub/repo_readme.png)

5. The README file is in Markdown — feel free to go wild with it! You can read more about writing good model cards [in our free course!](https://huggingface.co/course/chapter4/4?fw=pt)

If you look at the “Files and versions” tab, you’ll see that there aren’t many files there yet — just the README.md you just created and the .gitattributes file that keeps track of large files.


![/docs/assets/hub/files.png](/docs/assets/hub/files.png)


## Uploading your files

If you've used Git before, this will be very easy since Git is used to manage files in the Hub.

There is only one key difference if you have large files (over 10MB). These files are usually tracked with **git-lfs** (which stands for Git Large File Storage). 

1. Please make sure you have both **git** and **git-lfs** installed on your system.

* [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [Install Git LFS](https://git-lfs.github.com/)

2. Run `git lfs install` to initialize **git-lfs**:

```
git lfs install
```

3. Clone your model repository created in the previous section:

```
git clone https://huggingface.co/<your-username>/<your-model-id>
```

The directory should contain the `README.md` file created in the previous section.


4. Add your files to the repository

Now's the time 🔥. You can add any files you want to the repository. 

5. Commit and push your files

You can do this with the usual Git workflow

```
git add .
git commit -m "First model version"
git push
```

6. Inspect files and history

And we're done! You can check your repository with all the recently added files!

![/docs/assets/hub/repo_with_files.png](/docs/assets/hub/repo_with_files.png)

The UI allows you to explore the model files and commits and to see the diff introduced by each commit:

![/docs/assets/hub/explore_history.gif](/docs/assets/hub/explore_history.gif)

7. Add metadata

You can add metadata to your model card. You can specify:
* the type of task this model is for, enabling widgets and the Inference API.
* the used library (`transformers`, `spaCy`, etc)
* the language
* the dataset
* metrics
* license
* a lot more!

Read more about model tags [here](https://huggingface.co/docs/hub/model-repos#how-are-model-tags-determined).
