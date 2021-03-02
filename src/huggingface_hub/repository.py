import logging
import os
import re
import subprocess
from typing import List, Optional, Union

from .hf_api import HfFolder
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND


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

        If specifying a `clone_from`:
        will clone an existing remote repository, for instance one
        that was previously created using ``HfApi().create_repo(token=huggingface_token, name=repo_name)``.
        ``Repository`` uses the local git credentials by default, but if required, the ``huggingface_token``
        as well as the git ``user`` and the ``email`` can be explicitly specified.
        If `clone_from` is used, and the repository is being instantiated into a non-empty directory,
        e.g. a directory with your trained model files, it will automatically merge them.

        Args:
            local_dir (``str``):
                path (e.g. ``'my_trained_model/'``) to the local directory, where the ``Repository`` will be initalized.
            clone_from (``str``, optional):
                repository url (e.g. ``'https://huggingface.co/philschmid/playground-tests'``).
            use_auth_token (``str`` or ``bool``, `optional`, defaults ``None``):
                huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate against the hub
                (useful from Google Colab for instance).
            git_user (``str``, `optional`, defaults ``None``):
                will override the ``git config user.name`` for committing and pushing files to the hub.
            git_email (``str``, `optional`, defaults ``None``):
                will override the ``git config user.email`` for committing and pushing files to the hub.
        """

        os.makedirs(local_dir, exist_ok=True)
        self.local_dir = local_dir

        self.check_git_versions()

        if clone_from is not None:
            self.clone_from(repo_url=clone_from, use_auth_token=use_auth_token)
        else:
            if os.path.isdir(os.path.join(self.local_dir, ".git")):
                logger.debug("[Repository] is a valid git repo")
            else:
                logger.error(
                    "If not specifying `clone_from`, you need to pass Repository a valid git clone."
                )
                raise ValueError(
                    "If not specifying `clone_from`, you need to pass Repository a valid git clone."
                )

        # overrides .git config if user and email is provided.
        if git_user is not None or git_email is not None:
            self.git_config_username_and_email(git_user, git_email)

    def check_git_versions(self):
        """
        print git and git-lfs versions, raises if they aren't installed.
        """
        try:
            git_version = subprocess.run(
                ["git", "--version"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
            ).stdout.strip()
        except FileNotFoundError:
            raise EnvironmentError(
                "Looks like you do not have git installed, please install."
            )

        try:
            lfs_version = subprocess.run(
                ["git-lfs", "--version"],
                encoding="utf-8",
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            ).stdout.strip()
        except FileNotFoundError:
            raise EnvironmentError(
                "Looks like you do not have git-lfs installed, please install."
                " You can install from https://git-lfs.github.com/."
                " Then run `git lfs install` (you only have to do this once)."
            )
        logger.info(git_version + "\n" + lfs_version)

    def clone_from(self, repo_url: str, use_auth_token: Union[bool, str, None] = None):
        """
        Clone from a remote.
        """
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token:
            huggingface_token = HfFolder.get_token()
        else:
            huggingface_token = None

        if (
            huggingface_token is not None
            and "huggingface.co" in repo_url
            and "@" not in repo_url
        ):
            # adds huggingface_token to repo url if it is provided.
            # do not leak user token if it's not a repo on hf.co
            repo_url = repo_url.replace(
                "https://", f"https://user:{huggingface_token}@"
            )
        try:
            subprocess.run(
                "git lfs install".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
            )

            # checks if repository is initialized in a empty repository or in one with files
            if len(os.listdir(self.local_dir)) == 0:
                subprocess.run(
                    ["git", "clone", repo_url, "."],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
            else:
                logger.warning(
                    "[Repository] local_dir is not empty, so let's try to pull the remote over a non-empty folder."
                )
                subprocess.run(
                    "git init".split(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
                subprocess.run(
                    ["git", "remote", "add", "origin", repo_url],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
                subprocess.run(
                    "git fetch".split(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
                subprocess.run(
                    "git reset origin/main".split(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    encoding="utf-8",
                    check=True,
                    cwd=self.local_dir,
                )
                # TODO(check if we really want the --force flag)
                subprocess.run(
                    "git checkout origin/main -ft".split(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    encoding="utf-8",
                    check=True,
                    cwd=self.local_dir,
                )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_config_username_and_email(
        self, git_user: Optional[str] = None, git_email: Optional[str] = None
    ):
        """
        sets git user name and email (only in the current repo)
        """
        try:
            if git_user is not None:
                subprocess.run(
                    ["git", "config", "user.name", git_user],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
            if git_email is not None:
                subprocess.run(
                    ["git", "config", "user.email", git_email],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_head_hash(self) -> str:
        """
        Get commit sha on top of HEAD.
        """
        try:
            p = subprocess.run(
                "git rev-parse HEAD".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                encoding="utf-8",
                check=True,
                cwd=self.local_dir,
            )
            return p.stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_remote_url(self) -> str:
        """
        Get URL to origin remote.
        """
        try:
            p = subprocess.run(
                "git config --get remote.origin.url".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                encoding="utf-8",
                check=True,
                cwd=self.local_dir,
            )
            url = p.stdout.strip()
            # Strip basic auth info.
            return re.sub(r"https://.*@", "https://", url)
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_head_commit_url(self) -> str:
        """
        Get URL to last commit on HEAD
        We assume it's been pushed, and the url scheme is
        the same one as for GitHub or HuggingFace.
        """
        sha = self.git_head_hash()
        url = self.git_remote_url()
        if url.endswith("/"):
            url = url[:-1]
        return f"{url}/commit/{sha}"

    def lfs_track(self, patterns: Union[str, List[str]]):
        """
        Tell git-lfs to track those files.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        try:
            for pattern in patterns:
                subprocess.run(
                    ["git", "lfs", "track", pattern],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def lfs_enable_largefiles(self):
        """
        HF-specific. This enables upload support of files >5GB.
        """
        try:
            subprocess.run(
                "git config lfs.customtransfer.multipart.path huggingface-cli".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
            subprocess.run(
                f"git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_pull(self, rebase: Optional[bool] = False):
        """
        git pull
        """
        args = "git pull".split()
        if rebase:
            args.append("--rebase")
        try:
            subprocess.run(
                args,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_add(self, pattern="."):
        """
        git add
        """
        try:
            subprocess.run(
                ["git", "add", pattern],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_commit(self, commit_message="commit files to HF hub"):
        """
        git commit
        """
        try:
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
        except subprocess.CalledProcessError as exc:
            if len(exc.stderr) > 0:
                raise EnvironmentError(exc.stderr)
            else:
                raise EnvironmentError(exc.stdout)

    def git_push(self) -> str:
        """
        git push

        Returns url to commit on remote repo.
        """
        try:
            result = subprocess.run(
                "git push".split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
            logger.info(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

        return self.git_head_commit_url()

    def push_to_hub(self, commit_message="commit files to HF hub") -> str:
        """
        Helper to add, commit, and pushe file to remote repository on the HuggingFace Hub.
        Args:
            commit_message: commit message.
        """
        self.git_add()
        self.git_commit(commit_message)
        return self.git_push()
