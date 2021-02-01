import os
import subprocess


class HfRepository:
    """Git-based system for HuggingFace Hub repositories"""

    def __init__(
        self, repo_url: str, model_dir=".", huggingface_token: str = None, user: str = None, email: str = None,
    ):
        """
        Initializes an existing HuggingFace-Hub repository that was previously created using ``HfApi().create_repo(token=huggingface_token,name=repo_name)``.
        ``HfRepository`` uses the local git credentials by default, but if required, the ``huggingface_token`` as well as the git ``user`` and the ``email`` can be specified.
        ``HfRepository`` will then override them. If the repository is being initialized into a directory where files already exists, e.g. a directory with your
        trained model files, it will automatically merge them.
        Args:
            repo_url (``str``): repository url (e.g. ``'https://huggingface.co/philschmid/playground-tests'``) of the ``HfRepository`` on the HuggingFace Hub.
            model_dir (``str``, `optional`, defaults ``.``): path (e.g. ``'my_trained_model/'``) to the local directory, where the ``HfRepository``will be either cloned or initalized.
            huggingface_token (``str``, `optional`, defaults ``None``): huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate against the hub.
            user (``str``, `optional`, defaults ``None``): will override the ``git config user.name`` for committing and pushing files to the hub.
            email (``str``, `optional`, defaults ``None``): will override the ``git config user.email`` for committing and pushing files to the hub.
        """

        self.repo_url = repo_url
        if huggingface_token is not None:
            self.repo_url.replace("https://", f"https://user:{huggingface_token}@")

        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

        if len(os.listdir(model_dir)) == 0:
            subprocess.run(f"git clone {self.repo_url}".split(), check=True, cwd=self.model_dir)
        else:
            subprocess.run("git init".split(), check=True, cwd=self.model_dir)
            subprocess.run(
                f"git remote add origin {self.repo_url}".split(), check=True, cwd=self.model_dir,
            )
            subprocess.run("git fetch".split(), check=True, cwd=self.model_dir)
            subprocess.run("git reset origin/main".split(), check=True, cwd=self.model_dir)
            subprocess.run("git checkout origin/main -ft".split(), check=True, cwd=self.model_dir)

        subprocess.run("git lfs install".split(), check=True)

        if user is not None and email is not None:
            self.config_git_username_and_email(user, email)

    def config_git_username_and_email(self, user: str, email: str):
        """
        sets git user and email for commiting files to repository
        Args:
            user (``str``): will override the ``git config user.name`` for committing and pushing files to the hub.
            email (``str``): will override the ``git config user.email`` for committing and pushing files to the hub.
        """
        subprocess.run(f"git config user.email {email}".split(), check=True)
        subprocess.run(f"git config user.name {user}".split(), check=True)

    def commit_files(self, commit_message="commit files to HF hub"):
        """
        adds and commits files ine ``model_dir``.
        Args:
            commit_message (``str``, default ``'commit files to HF hub'``): commit message.
        """
        subprocess.run("git add .".split(), check=True, cwd=self.model_dir)
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=self.model_dir)

    def push_files(self):
        """
        pushes committed files to remote repository on the HuggingFace Hub.
        """
        subprocess.run("git push".split(), check=True, cwd=self.model_dir)

    def push_to_hub(self, commit_message="commit files to HF hub"):
        """
        commits and pushed files to remote repository on the HuggingFace Hub.
        Args:
            commit_message (``str``, default ``'commit files to HF hub'``): commit message.
        """
        self.commit_files(commit_message)
        self.push_files()
