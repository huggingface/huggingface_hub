import logging
import os
import subprocess
from typing import Optional

from .constants import HUGGINGFACE_CO_URL_HOME


logger = logging.getLogger(__name__)


class Repository:
    """
    Helper class to wrap the git and git-lfs commands.

    The aim is to facilitate interacting with huggingface.co hosted model or dataset repos,
    though not a lot here (if any) is actually specific to huggingface.co.
    """

    def __init__(
        self,
        local_dir: str,
        clone_from: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
    ):
        """
        Instantiate a local clone of a git repo.

        If specifying a `clone_from`, will clone an existing HuggingFace-Hub repository
        that was previously created using ``HfApi().create_repo(token=huggingface_token, name=repo_name)``.
        ``Repository`` uses the local git credentials by default, but if required, the ``huggingface_token``
        as well as the git ``user`` and the ``email`` can be specified.
        ``Repository`` will then override them.
        If `clone_from` is used, and the repository is being instantiated into a directory where files already exists,
        e.g. a directory with your trained model files, it will automatically merge them.

        Args:
            local_dir (``str``):
                path (e.g. ``'my_trained_model/'``) to the local directory, where the ``Repository`` will be either initalized.
            repo_url (``str``, optional):
                repository url (e.g. ``'https://huggingface.co/philschmid/playground-tests'``).
            use_auth_token (``str`` or ``bool``, `optional`, defaults ``None``):
                huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate against the hub.
            git_user (``str``, `optional`, defaults ``None``):
                will override the ``git config user.name`` for committing and pushing files to the hub.
            git_email (``str``, `optional`, defaults ``None``):
                will override the ``git config user.email`` for committing and pushing files to the hub.
        """

        os.makedirs(local_dir, exist_ok=True)
        self.local_dir = local_dir

        self.check_git_versions()

        if clone_from:
            self.clone_from(repo_url=clone_from, use_auth_token=use_auth_token)
        else:
            subprocess.run("git remote -v")
            if error:
                raise
            else:
                pass

        # overrides .git config if user and email is provided.
        if user is not None or email is not None:
            self.git_config_username_and_email(user, email)

    def check_git_versions(self):
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

        try:
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

    def clone_from(self, repo_url: str, use_auth_token: Union[bool, str, None] = None):

        # adds huggingface_token to repo url if it is provided.
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token:
            huggingface_token = HfFolder.get_token()

        if huggingface_token is not None and repo_url.startswith(
            HUGGINGFACE_CO_URL_HOME
        ):
            repo_url = repo_url.replace(
                "https://", f"https://user:{huggingface_token}@"
            )

        subprocess.run("git lfs install".split(), check=True)

        # checks if repository is initialized in a empty repository or in one with files
        if len(os.listdir(self.local_dir)) == 0:
            subprocess.run(
                f"git clone {self.repo_url}".split(), check=True, cwd=self.local_dir
            )
        else:
            logger.warning(
                "[Repository] local_dir is not empty, so let's try to pull the remote over a non-empty folder."
            )
            subprocess.run("git init".split(), check=True, cwd=self.local_dir)
            subprocess.run(
                f"git remote add origin {self.repo_url}".split(),
                check=True,
                cwd=self.local_dir,
            )
            subprocess.run("git fetch".split(), check=True, cwd=self.local_dir)
            subprocess.run(
                "git reset origin/main".split(), check=True, cwd=self.local_dir
            )
            # TODO(check if we really want the --force flag)
            subprocess.run(
                "git checkout origin/main -ft".split(), check=True, cwd=self.local_dir
            )

    def git_config_username_and_email(
        self, user: Optional[str] = None, email: Optional[str] = None
    ):
        """
        sets git user and email for committing files to repository
        Args:
            user (``str``):
                will override the ``git config user.name`` for committing.
            email (``str``):
                will override the ``git config user.email`` for committing.
        """
        if email is not None:
            subprocess.run(
                f"git config user.email {email}".split(), check=True, cwd=self.local_dir
            )
        if user is not None:
            subprocess.run(
                f"git config user.name {user}".split(), check=True, cwd=self.local_dir
            )

    def git_lfs_track():
        pass

    def git_add(self):
        """
        adds and commits files ine ``local_dir``.
        Args:
            commit_message (``str``, default ``'commit files to HF hub'``): commit message.
        """
        subprocess.run("git add .".split(), check=True, cwd=self.local_dir)
        subprocess.run(
            ["git", "commit", "-m", commit_message], check=True, cwd=self.local_dir
        )

    def git_push(self):
        """
        pushes committed files to remote repository on the HuggingFace Hub.
        """
        subprocess.run("git push".split(), check=True, cwd=self.local_dir)

    def commit_files(self, commit_message="commit files to HF hub"):
        """
        adds and commits files ine ``local_dir``.
        Args:
            commit_message (``str``, default ``'commit files to HF hub'``): commit message.
        """
        subprocess.run("git add .".split(), check=True, cwd=self.local_dir)
        subprocess.run(
            ["git", "commit", "-m", commit_message], check=True, cwd=self.local_dir
        )

    def push_to_hub(self, commit_message="commit files to HF hub"):
        """
        commits and pushed files to remote repository on the HuggingFace Hub.
        Args:
            commit_message (``str``, default ``'commit files to HF hub'``): commit message.
        """
        self.commit_files(commit_message)
        self.push_files()
