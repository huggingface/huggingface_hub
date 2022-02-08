---
title: Adding your model to the Hugging Face Hub
---

# Adding your model to the Hugging Face Hub

## Why?

Uploading models to the Hugging Face Hub has many [benefits](https://huggingface.co/docs/hub/main#whats-a-repository). In summary, you get versioning, branches, discoverability and sharing features, integration with over a dozen libraries and more!

## Accounts and organizations

The first step is to create an account at [Hugging Face](https://huggingface.co/login). The models are shared in the form of Git-based repositories. You have control over your repository, so you can have checkpoints, configs and any files you might want to upload.

The repository can be either linked with an individual, such as [osanseviero/fashion_brands_patterns](https://huggingface.co/osanseviero/fashion_brands_patterns) or with an organization, such as [facebook/bart-large-xsum](https://huggingface.co/facebook/bart-large-xsum). Organizations can be used if you want to upload models that are related to a company, community or library! If you choose an organization, the model will be featured on the organization’s page and every member of the organization will have the ability to contribute to the repository. You can create a new organization [here](https://huggingface.co/organizations/new).

## How is this tutorial written?

This tutorial is split into three parts:
- [Using the web interface to create a model repository and upload your model (Beginner)](##using-the-web-interface)
- [Using the web interface to create a model repository and using the `huggingface-cli` to upload your model (Intermediate)](##using-the-web-interface-and-command-line)
- [Using the `huggingface_hub` library to do so entirely from your Python interface (Advanced)](##using-the-huggingface_hub-client-library)

Each section will cover uploading the same model to the same repository with a different method. Each method has different advantages and disadvantages depending on your use case.

## Using the Web Interface

### Creating a Repository

First, follow the previous [Creating a Repository](##using-the-web-interface) directions to create one from the web interface.

### Uploading your Model

1. In the "Files and versions" tab, select "Add File" and specify "Upload File":

![/docs/assets/hub/add-file.png](docs/assets/hub/add-file.png)

2. From there, select a file from your computer to upload and leave a helpful commit message to know what you are uploading:

![docs/assets/hub/commit-file.png](/docs/assets/hub/commit-file.png)

3. Afterwards hit "Commit changes" and your model will be uploaded to the Hub!

4. Inspect files and history

You can check your repository with all the recently added files!

![/docs/assets/hub/repo_with_files.png](/docs/assets/hub/repo_with_files.png)

The UI allows you to explore the model files and commits and to see the diff introduced by each commit:

![/docs/assets/hub/explore_history.gif](/docs/assets/hub/explore_history.gif)

5. Add metadata

You can add metadata to your model card. You can specify:
* the type of task this model is for, enabling widgets and the Inference API.
* the used library (`transformers`, `spaCy`, etc)
* the language
* the dataset
* metrics
* license
* a lot more!

Read more about model tags [here](/docs/hub/model-repos#model-card-metadata).

6. Add TensorBoard traces



Any repository that contains TensorBoard traces (filenames that contain `tfevents`) is categorized with the [`TensorBoard` tag](https://huggingface.co/models?filter=tensorboard). As a convention, we suggest that you save traces under the `runs/` subfolder. The "Training metrics" tab then makes it easy to review charts of the logged variables, like the loss or the accuracy.

![Training metrics tab on a model's page, with TensorBoard](/docs/assets/hub/tensorboard.png)

Models trained with 🤗 Transformers will generate [TensorBoard traces](https://huggingface.co/transformers/main_classes/callback.html?highlight=tensorboard#transformers.integrations.TensorBoardCallback) by default if [`tensorboard`](https://pypi.org/project/tensorboard/) is installed.


## Using the Web Interface and Command Line

### Creating a repository

Using the web interface, you can easily create repositories, add files (even large ones!), explore models, visualize diffs, and much more. Let's begin by creating a repository.

1. To create a new repository, visit [huggingface.co/new](http://huggingface.co/new):

![/docs/assets/hub/new_repo.png](/docs/assets/hub/new_repo.png)

2. First, specify the owner of the repository: this can be either you or any of the organizations you’re affiliated with. 

3. Next, enter your model’s name. This will also be the name of the repository. Finally, you can specify whether you want your model to be public or private.

After creating your model repository, you should see a page like this:

![/docs/assets/hub/empty_repo.png](/docs/assets/hub/empty_repo.png)

4. This is where your model will be hosted. For now, only the README.md file will be in there. It's in Markdown — feel free to go wild with it! You can read more about writing good model cards [in our free course!](https://huggingface.co/course/chapter4/4?fw=pt)

If you look at the “Files and versions” tab, you’ll see that there aren’t many files there yet — just the README.md you just created and the .gitattributes file that keeps track of large files.


![/docs/assets/hub/files.png](/docs/assets/hub/files.png)


### Uploading your files

If you've used Git before, this will be very easy since Git is used to manage files in the Hub.

There is only one key difference if you have large files (over 10MB, or 1MB for binary files). These files are usually tracked with **git-lfs** (which stands for Git Large File Storage). 

1. Please make sure you have both **git** and **git-lfs** installed on your system.

* [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [Install Git LFS](https://git-lfs.github.com/)

2. Run `git lfs install` to initialize **git-lfs**:

Do you have files larger than 10MB? Those files are tracked with `git-lfs`. We already provide a list of common file extensions for these files in `.gitattributes`, but you might need to add new extensions if they are not already handled. You can do so with `git lfs track "*.your_extension"`.

Once ready, just run:

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

You can do this with the usual Git workflow:

```
git add .
git commit -m "First model version"
git push
```
And we're done!

You can check your repository with all the recently added files!

![/docs/assets/hub/repo_with_files.png](/docs/assets/hub/repo_with_files.png)

The UI allows you to explore the model files and commits and to see the diff introduced by each commit:

![/docs/assets/hub/explore_history.gif](/docs/assets/hub/explore_history.gif)


## Using the `huggingface_hub` client library

### Installing the library and logging in

Before we begin, you should make sure the `huggingface_hub` library is installed on your system by running the following `bash` command:

>>> pip install huggingface_hub

Afterwards, you should login with your credentials. To find your credentials you can go to your [settings](https://huggingface.co/settings/token) from the Hugging Face website and copy the current token there.

To login to your profile, there are two options:
- Logging in from a Jupyter Notebook that has ipywidgets enabled
- Logging in from the command line

To login through Jupyter, run the following inside of a Notebook cell:

>>> from huggingface_hub import notebook_login
>>> notebook_login()

You will be presented with a prompt similar to the following, where it will ask you to paste in that login token from earlier

![/docs/assets/hub/notebook_login.png](/docs/assets/hub/notebook_login.png)

To login through the command line, run the following from a terminal:

>>> huggingface-cli login

You will be presented with a prompt asking for you to paste your token
![/docs/assets/hub/cli-login.png](/docs/assets/hub/cli-login.png)

After either login method is chosen, you will be asked to run:
>>> git config --global credential.helper store

This ensures that git is looking at our newly-stored credentials any time we wish to push to the Hub

> Note: You may find that `HfApi` has a `set_access_token` function. This does not set all the permissions needed at each location, and is more for internal use. You should use one of the two methods mentioned above.

### Creating a repository

When using the `huggingface_hub`, we can create a new repository from just a few lines of code!
First we need to instantiate the `HfApi` class, which holds all of the magic:
>>> from huggingface_hub import HfApi
>>> api = HfApi()

Afterwards we can run the `create_repo` function, specifying a number of settings and options for our new repository:
>>> api.create_repo(
>>>   name = "dummy", # The name of our repository
>>>   organization = None, # The namespace of the expected repository. Automatically grabs your logged-in profile name
>>>   private = False, # Whether the repo should be public or private
>>>   repo_type = "model" # The type of repository, such as "model", "space", "dataset"
>>> )

> To read more about what you can pass in, check out its documentation by doing api.create_repo?

### Uploading your files

There are two methods for uploading a file to the Hub:
- `HfApi.upload_file`
- `Repository.push_to_hub`

`upload_file` should be used when the file is quite small (less than 10MB), and is straightforward to use. Simply pass in the filename, the location it should be in the repository, and the name of the repository.

In this example we'll write a quick `README.md` file: 
>>> with open('README.md', 'w+') as f:
>>>     f.write('''# Dummy model
>>>     
>>> This is a dummy model''')

And quickly push it to the Hub:
>>> url = api.upload_file(
>>>    path_or_fileobj = 'README.md', 
>>>    path_in_repo = 'README.md', 
>>>    repo_id = 'my_username/dummy',
>>>)

You can find your file live on the Hub at the url returned from `upload_file`


If you are trying to upload larger files to the hub (over 10MB), you should ensure that **git-lfs** is installed on your system. Git is used to manage your files on the Hub, and tracking of large file storages needs to utilize this. 

First we need to clone our repository from the Hub by doing:
>>> from huggingface_hub import Repository
>>> repo = Repository(
>>>    local_dir = 'dummy', 
>>>    clone_from='my_username/dummy'
>>> )

Then we can write to our `dummy` folder any large file we may want to store, before finally pushing to the hub with `Repository.push_to_hub`, attaching a helpeful commit message to it:

>>> repo.push_to_hub(
>>>   commit_message = "Our first big model!"
>>> )

And that's it! You can now push your models and files to your newly created Repository without ever having to leave your Python interpreter.

You can check now check your repository with all the recently added files:

![/docs/assets/hub/repo_with_files.png](/docs/assets/hub/repo_with_files.png)

The UI allows you to explore the model files and commits and to see the diff introduced by each commit:

![/docs/assets/hub/explore_history.gif](/docs/assets/hub/explore_history.gif)


