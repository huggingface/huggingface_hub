import atexit
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES, REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save

from .hf_api import ENDPOINT, HfApi, HfFolder, repo_type_and_id_from_hf_id
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND
from .utils import logging


logger = logging.get_logger(__name__)


class CommandInProgress:
    def __init__(
        self,
        title: str,
        is_done_method: Callable,
        status_method: Callable,
        process: subprocess.Popen,
        post_method: Optional[Callable] = None,
    ):
        self.title = title
        self._is_done = is_done_method
        self._status = status_method
        self._process = process
        self._stderr = ""
        self._stdout = ""
        self._post_method = post_method

    @property
    def is_done(self) -> bool:
        """
        Whether the process is done.
        """
        result = self._is_done()

        if result and self._post_method is not None:
            self._post_method()

        return result

    @property
    def status(self) -> int:
        """
        The exit code/status of the current action. Will return `0` if the command has completed
        successfully, and a number between 1 and 255 if the process errored-out.

        Will return -1 if the command is still ongoing.
        """
        return self._status()

    @property
    def failed(self) -> bool:
        """
        Whether the process errored-out.
        """
        return self.status > 0

    @property
    def stderr(self) -> str:
        """
        The current output message on the standard error.
        """
        self._stderr += self._process.stderr.read()
        return self._stderr

    @property
    def stdout(self) -> str:
        """
        The current output message on the standard output.
        """
        self._stdout += self._process.stdout.read()
        return self._stdout

    def __repr__(self):
        status = self.status

        if status == -1:
            status = "running"

        return f"[{self.title} command, status code: {status}, {'in progress.' if not self.is_done else 'finished.'} PID: {self._process.pid}]"


def is_git_repo(folder: Union[str, Path]) -> bool:
    """
    Check if the folder is the root of a git repository
    """
    folder_exists = os.path.exists(os.path.join(folder, ".git"))
    git_branch = subprocess.run(
        "git branch".split(), cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return folder_exists and git_branch.returncode == 0


def is_local_clone(folder: Union[str, Path], remote_url: str) -> bool:
    """
    Check if the folder is the a local clone of the remote_url
    """
    if not is_git_repo(folder):
        return False

    remotes = subprocess.run(
        "git remote -v".split(),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=folder,
    ).stdout

    # Remove token for the test with remotes.
    remote_url = re.sub(r"https://.*@", "https://", remote_url)
    remotes = [re.sub(r"https://.*@", "https://", remote) for remote in remotes.split()]
    return remote_url in remotes


def is_tracked_with_lfs(filename: Union[str, Path]) -> bool:
    """
    Check if the file passed is tracked with git-lfs.
    """
    folder = Path(filename).parent
    filename = Path(filename).name

    try:
        p = subprocess.run(
            ["git", "check-attr", "-a", filename],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=folder,
        )
        attributes = p.stdout.strip()
    except subprocess.CalledProcessError as exc:
        if not is_git_repo(folder):
            return False
        else:
            raise OSError(exc.stderr)

    if len(attributes) == 0:
        return False

    found_lfs_tag = {"diff": False, "merge": False, "filter": False}

    for attribute in attributes.split("\n"):
        for tag in found_lfs_tag.keys():
            if tag in attribute and "lfs" in attribute:
                found_lfs_tag[tag] = True

    return all(found_lfs_tag.values())


def is_git_ignored(filename: Union[str, Path]) -> bool:
    """
    Check if file is git-ignored. Supports nested .gitignore files.
    """
    folder = Path(filename).parent
    filename = Path(filename).name

    try:
        p = subprocess.run(
            ["git", "check-ignore", filename],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            cwd=folder,
        )
        # Will return exit code 1 if not gitignored
        is_ignored = not bool(p.returncode)
    except subprocess.CalledProcessError as exc:
        raise OSError(exc.stderr)

    return is_ignored


def files_to_be_staged(pattern: str, folder: Union[str, Path]) -> List[str]:
    try:
        p = subprocess.run(
            ["git", "ls-files", "-mo", pattern],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=folder,
        )
        if len(p.stdout.strip()):
            files = p.stdout.strip().split("\n")
        else:
            files = []
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return files


def is_tracked_upstream(folder: Union[str, Path]) -> bool:
    """
    Check if the current checked-out branch is tracked upstream.
    """
    try:
        command = "git rev-parse --symbolic-full-name --abbrev-ref @{u}"
        subprocess.run(
            command.split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            check=True,
            cwd=folder,
        )
        return True
    except subprocess.CalledProcessError as exc:
        if "HEAD" in exc.stderr:
            raise OSError("No branch checked out")

        return False


def commits_to_push(folder: Union[str, Path], upstream: Optional[str] = None) -> int:
    """
    Check the number of commits that would be pushed upstream
    """
    try:
        command = f"git cherry -v {upstream or ''}"
        result = subprocess.run(
            command.split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            check=True,
            cwd=folder,
        )
        return len(result.stdout.split("\n")) - 1
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)


@contextmanager
def lfs_log_progress():
    """
    This is a context manager that will log the Git LFS progress of cleaning, smudging, pulling and pushing.
    """

    if logger.getEffectiveLevel() >= logging.ERROR:
        try:
            yield
        finally:
            return

    def output_progress(stopping_event: threading.Event):
        """
        To be launched as a separate thread with an event meaning it should stop the tail.
        """
        pbars = {}

        def close_pbars():
            for pbar in pbars.values():
                pbar["bar"].update(pbar["bar"].total - pbar["past_bytes"])
                pbar["bar"].refresh()
                pbar["bar"].close()

        def tail_file(filename) -> Iterator[str]:
            """
            Creates a generator to be iterated through, which will return each line one by one.
            Will stop tailing the file if the stopping_event is set.
            """
            with open(filename, "r") as file:
                current_line = ""
                while True:
                    if stopping_event.is_set():
                        close_pbars()
                        break

                    line_bit = file.readline()
                    if line_bit is not None and not len(line_bit.strip()) == 0:
                        current_line += line_bit
                        if current_line.endswith("\n"):
                            yield current_line
                            current_line = ""
                    else:
                        time.sleep(1)

        # If the file isn't created yet, wait for a few seconds before trying again.
        # Can be interrupted with the stopping_event.
        while not os.path.exists(os.environ["GIT_LFS_PROGRESS"]):
            if stopping_event.is_set():
                close_pbars()
                return

            time.sleep(2)

        for line in tail_file(os.environ["GIT_LFS_PROGRESS"]):
            state, file_progress, byte_progress, filename = line.split()
            description = f"{state.capitalize()} file {filename}"

            current_bytes, total_bytes = byte_progress.split("/")

            current_bytes = int(current_bytes)
            total_bytes = int(total_bytes)

            if pbars.get((state, filename)) is None:
                pbars[(state, filename)] = {
                    "bar": tqdm(
                        desc=description,
                        initial=current_bytes,
                        total=total_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ),
                    "past_bytes": current_bytes,
                }
            else:
                past_bytes = pbars[(state, filename)]["past_bytes"]
                pbars[(state, filename)]["bar"].update(current_bytes - past_bytes)
                pbars[(state, filename)]["past_bytes"] = current_bytes

    current_lfs_progress_value = os.environ.get("GIT_LFS_PROGRESS", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["GIT_LFS_PROGRESS"] = os.path.join(tmpdir, "lfs_progress")
        logger.debug(f"Following progress in {os.environ['GIT_LFS_PROGRESS']}")

        exit_event = threading.Event()
        x = threading.Thread(target=output_progress, args=(exit_event,), daemon=True)
        x.start()

        try:
            yield
        finally:
            exit_event.set()
            x.join()

            os.environ["GIT_LFS_PROGRESS"] = current_lfs_progress_value


class Repository:
    """
    Helper class to wrap the git and git-lfs commands.

    The aim is to facilitate interacting with huggingface.co hosted model or dataset repos,
    though not a lot here (if any) is actually specific to huggingface.co.
    """

    command_queue: List[CommandInProgress]

    def __init__(
        self,
        local_dir: str,
        clone_from: Optional[str] = None,
        repo_type: Optional[str] = None,
        use_auth_token: Union[bool, str] = True,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
        revision: Optional[str] = None,
        private: bool = False,
        skip_lfs_files: bool = False,
    ):
        """
        Instantiate a local clone of a git repo.

        If specifying a `clone_from`:
        will clone an existing remote repository, for instance one
        that was previously created using ``HfApi().create_repo(name=repo_name)``.
        ``Repository`` uses the local git credentials by default, but if required, the ``huggingface_token``
        as well as the git ``user`` and the ``email`` can be explicitly specified.
        If `clone_from` is used, and the repository is being instantiated into a non-empty directory,
        e.g. a directory with your trained model files, it will automatically merge them.

        Args:
            local_dir (``str``):
                path (e.g. ``'my_trained_model/'``) to the local directory, where the ``Repository`` will be initalized.
            clone_from (``str``, `optional`):
                repository url (e.g. ``'https://huggingface.co/philschmid/playground-tests'``).
            repo_type (``str``, `optional`):
                To set when creating a repo: et to "dataset" or "space" if creating a dataset or space, default is model.
            use_auth_token (``str`` or ``bool``, `optional`, defaults to ``True``):
                huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate against the hub
                (useful from Google Colab for instance).
            git_user (``str``, `optional`):
                will override the ``git config user.name`` for committing and pushing files to the hub.
            git_email (``str``, `optional`):
                will override the ``git config user.email`` for committing and pushing files to the hub.
            revision (``str``, `optional`):
                Revision to checkout after initializing the repository. If the revision doesn't exist, a
                branch will be created with that revision name from the default branch's current HEAD.
            private (``bool``, `optional`, defaults to ``False``):
                whether the repository is private or not.
            skip_lfs_files (``bool``, `optional`, defaults to ``False``):
                whether to skip git-LFS files or not.
        """

        os.makedirs(local_dir, exist_ok=True)
        self.local_dir = os.path.join(os.getcwd(), local_dir)
        self.repo_type = repo_type
        self.command_queue = []
        self.private = private
        self.skip_lfs_files = skip_lfs_files

        self.check_git_versions()

        if isinstance(use_auth_token, str):
            self.huggingface_token = use_auth_token
        elif use_auth_token:
            self.huggingface_token = HfFolder.get_token()
        else:
            self.huggingface_token = None

        if clone_from is not None:
            self.clone_from(repo_url=clone_from)
        else:
            if is_git_repo(self.local_dir):
                logger.debug("[Repository] is a valid git repo")
            else:
                raise ValueError(
                    "If not specifying `clone_from`, you need to pass Repository a valid git clone."
                )

        if self.huggingface_token is not None and (
            git_email is None or git_user is None
        ):
            user = HfApi().whoami(self.huggingface_token)

            if git_email is None:
                git_email = user["email"]

            if git_user is None:
                git_user = user["fullname"]

        if git_user is not None or git_email is not None:
            self.git_config_username_and_email(git_user, git_email)

        self.lfs_enable_largefiles()
        self.git_credential_helper_store()

        if revision is not None:
            self.git_checkout(revision, create_branch_ok=True)

        # This ensures that all commands exit before exiting the Python runtime.
        # This will ensure all pushes register on the hub, even if other errors happen in subsequent operations.
        atexit.register(self.wait_for_commands)

    @property
    def current_branch(self):
        """
        Returns the current checked out branch.
        """
        command = "git rev-parse --abbrev-ref HEAD"
        try:
            result = subprocess.run(
                command.split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

        return result

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
        Clone from a remote. If the folder already exists, will try to clone the repository within it.

        If this folder is a git repository with linked history, will try to update the repository.
        """
        token = use_auth_token if use_auth_token is not None else self.huggingface_token
        if token is None and self.private:
            raise ValueError(
                "Couldn't load Hugging Face Authorization Token. Credentials are required to work with private repositories."
                " Please login in using `huggingface-cli login` or provide your token manually with the `use_auth_token` key."
            )
        api = HfApi()

        if "huggingface.co" in repo_url or (
            "http" not in repo_url and len(repo_url.split("/")) <= 2
        ):
            repo_type, namespace, repo_id = repo_type_and_id_from_hf_id(repo_url)

            if repo_type is not None:
                self.repo_type = repo_type

            repo_url = ENDPOINT + "/"

            if self.repo_type in REPO_TYPES_URL_PREFIXES:
                repo_url += REPO_TYPES_URL_PREFIXES[self.repo_type]

            if token is not None:
                whoami_info = api.whoami(token)
                user = whoami_info["name"]
                valid_organisations = [org["name"] for org in whoami_info["orgs"]]

                if namespace is not None:
                    repo_url += f"{namespace}/"
                repo_url += repo_id

                repo_url = repo_url.replace("https://", f"https://user:{token}@")

                if namespace == user or namespace in valid_organisations:
                    api.create_repo(
                        repo_id,
                        token=token,
                        repo_type=self.repo_type,
                        organization=namespace,
                        exist_ok=True,
                        private=self.private,
                    )
            else:
                if namespace is not None:
                    repo_url += f"{namespace}/"
                repo_url += repo_id

        # For error messages, it's cleaner to show the repo url without the token.
        clean_repo_url = re.sub(r"https://.*@", "https://", repo_url)
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
                logger.warning(f"Cloning {clean_repo_url} into local empty directory.")

                with lfs_log_progress():
                    env = os.environ.copy()

                    if self.skip_lfs_files:
                        env.update({"GIT_LFS_SKIP_SMUDGE": "1"})

                    subprocess.run(
                        f"{'git clone' if self.skip_lfs_files else 'git lfs clone'} {repo_url} .".split(),
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        check=True,
                        encoding="utf-8",
                        cwd=self.local_dir,
                        env=env,
                    )
            else:
                # Check if the folder is the root of a git repository
                in_repository = is_git_repo(self.local_dir)

                if in_repository:
                    if is_local_clone(self.local_dir, repo_url):
                        logger.warning(
                            f"{self.local_dir} is already a clone of {clean_repo_url}. Make sure you pull the latest "
                            "changes with `repo.git_pull()`."
                        )
                    else:
                        output = subprocess.run(
                            "git remote get-url origin".split(),
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            encoding="utf-8",
                            cwd=self.local_dir,
                        )

                        error_msg = (
                            f"Tried to clone {clean_repo_url} in an unrelated git repository.\nIf you believe this is "
                            f"an error, please add a remote with the following URL: {clean_repo_url}."
                        )
                        if output.returncode == 0:
                            clean_local_remote_url = re.sub(
                                r"https://.*@", "https://", output.stdout
                            )
                            error_msg += f"\nLocal path has its origin defined as: {clean_local_remote_url}"

                        raise EnvironmentError(error_msg)

                if not in_repository:
                    raise EnvironmentError(
                        "Tried to clone a repository in a non-empty folder that isn't a git repository. If you really "
                        "want to do this, do it manually:\n"
                        "git init && git remote add origin && git pull origin main\n"
                        " or clone repo to a new folder and move your existing files there afterwards."
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

    def git_credential_helper_store(self):
        """
        sets the git credential helper to `store`
        """
        try:
            subprocess.run(
                ["git", "config", "credential.helper", "store"],
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

    def list_deleted_files(self) -> List[str]:
        """
        Returns a list of the files that are deleted in the working directory or index.
        """
        try:
            git_status = subprocess.run(
                ["git", "status", "-s"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

        if len(git_status) == 0:
            return []

        # Receives a status like the following
        #  D .gitignore
        #  D new_file.json
        # AD new_file1.json
        # ?? new_file2.json
        # ?? new_file4.json

        # Strip each line of whitespaces
        modified_files_statuses = [status.strip() for status in git_status.split("\n")]

        # Only keep files that are deleted using the D prefix
        deleted_files_statuses = [
            status for status in modified_files_statuses if "D" in status.split()[0]
        ]

        # Remove the D prefix and strip to keep only the relevant filename
        deleted_files = [
            status.split()[-1].strip() for status in deleted_files_statuses
        ]

        return deleted_files

    def lfs_track(
        self, patterns: Union[str, List[str]], filename: Optional[bool] = False
    ):
        """
        Tell git-lfs to track those files.

        Setting the `filename` argument to `True` will treat the arguments as literal filenames,
        not as patterns. Any special glob characters in the filename will be escaped when
        writing to the .gitattributes file.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        try:
            for pattern in patterns:
                cmd = f"git lfs track {'--filename' if filename else ''} {pattern}"
                subprocess.run(
                    cmd.split(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def lfs_untrack(self, patterns: Union[str, List[str]]):
        """
        Tell git-lfs to untrack those files.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        try:
            for pattern in patterns:
                subprocess.run(
                    ["git", "lfs", "untrack", pattern],
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

    def auto_track_large_files(self, pattern: Optional[str] = ".") -> List[str]:
        """
        Automatically track large files with git-lfs
        """
        files_to_be_tracked_with_lfs = []

        deleted_files = self.list_deleted_files()

        for filename in files_to_be_staged(pattern, folder=self.local_dir):
            if filename in deleted_files:
                continue

            path_to_file = os.path.join(os.getcwd(), self.local_dir, filename)
            size_in_mb = os.path.getsize(path_to_file) / (1024 * 1024)

            if (
                size_in_mb >= 10
                and not is_tracked_with_lfs(path_to_file)
                and not is_git_ignored(path_to_file)
            ):
                self.lfs_track(filename)
                files_to_be_tracked_with_lfs.append(filename)

        # Cleanup the .gitattributes if files were deleted
        self.lfs_untrack(deleted_files)

        return files_to_be_tracked_with_lfs

    def lfs_prune(self, recent=False):
        """
        git lfs prune
        """
        args = "git lfs prune".split()
        if recent:
            args.append("--recent")
        try:
            with lfs_log_progress():
                result = subprocess.run(
                    args,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
                logger.info(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_pull(self, rebase: Optional[bool] = False, lfs: Optional[bool] = False):
        """
        git pull

        Args:
            rebase (`bool`, defaults to `False`):
                Whether to rebase the current branch on top of the upstream branch after fetching.
            lfs (`bool`, defaults to `False`):
                Whether to fetch the LFS files too. This option only changes the behavior when a repository
                was cloned without fetching the LFS files; calling `repo.git_pull(lfs=True)` will then fetch
                the LFS file from the remote repository.
        """
        args = ("git pull" if not lfs else "git lfs pull").split()
        if rebase:
            args.append("--rebase")
        try:
            with lfs_log_progress():
                result = subprocess.run(
                    args,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )
                logger.info(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_add(
        self, pattern: Optional[str] = ".", auto_lfs_track: Optional[bool] = False
    ):
        """
        git add

        Setting the `auto_lfs_track` parameter to `True` will automatically track files that are larger
        than 10MB with `git-lfs`.
        """
        if auto_lfs_track:
            tracked_files = self.auto_track_large_files(pattern)
            if tracked_files:
                logger.warning(
                    f"Adding files tracked by Git LFS: {tracked_files}. This may take a bit of time if the files are large."
                )

        try:
            result = subprocess.run(
                ["git", "add", "-v", pattern],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
            logger.info(f"Adding to index:\n{result.stdout}\n")
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

    def git_commit(self, commit_message: str = "commit files to HF hub"):
        """
        git commit
        """
        try:
            result = subprocess.run(
                ["git", "commit", "-m", commit_message, "-v"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
            logger.info(f"Committed:\n{result.stdout}\n")
        except subprocess.CalledProcessError as exc:
            if len(exc.stderr) > 0:
                raise EnvironmentError(exc.stderr)
            else:
                raise EnvironmentError(exc.stdout)

    def git_push(
        self,
        upstream: Optional[str] = None,
        blocking: Optional[bool] = True,
        auto_lfs_prune: Optional[bool] = False,
    ) -> Union[str, Tuple[str, CommandInProgress]]:
        """
        git push

        If used without setting `blocking`, will return url to commit on remote repo.
        If used with `blocking=True`, will return a tuple containing the url to commit
        and the command object to follow for information about the process.

        Args:
            upstream (`str`, `optional`):
                Upstream to which this should push. If not specified, will push
                to the lastly defined upstream or to the default one (`origin main`).
            blocking (`bool`, defaults to `True`):
                Whether the function should return only when the push has finished.
                Setting this to `False` will return an `CommandInProgress` object
                which has an `is_done` property. This property will be set to
                `True` when the push is finished.
            auto_lfs_prune (`bool`, defaults to `False`):
                Whether to automatically prune files once they have been pushed to the remote.
        """
        command = "git push"

        if upstream:
            command += f" --set-upstream {upstream}"

        number_of_commits = commits_to_push(self.local_dir, upstream)

        if number_of_commits > 1:
            logger.warning(
                f"Several commits ({number_of_commits}) will be pushed upstream."
            )
            if blocking:
                logger.warning("The progress bars may be unreliable.")

        try:
            with lfs_log_progress():
                process = subprocess.Popen(
                    command.split(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    encoding="utf-8",
                    cwd=self.local_dir,
                )

                if blocking:
                    stdout, stderr = process.communicate()
                    return_code = process.poll()
                    process.kill()

                    if len(stderr):
                        logger.warning(stderr)

                    if return_code:
                        raise subprocess.CalledProcessError(
                            return_code, process.args, output=stdout, stderr=stderr
                        )

        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

        if not blocking:

            def status_method():
                status = process.poll()
                if status is None:
                    return -1
                else:
                    return status

            command = CommandInProgress(
                "push",
                is_done_method=lambda: process.poll() is not None,
                status_method=status_method,
                process=process,
                post_method=self.lfs_prune if auto_lfs_prune else None,
            )

            self.command_queue.append(command)

            return self.git_head_commit_url(), command

        if auto_lfs_prune:
            self.lfs_prune()

        return self.git_head_commit_url()

    def git_checkout(self, revision: str, create_branch_ok: Optional[bool] = False):
        """
        git checkout a given revision

        Specifying `create_branch_ok` to `True` will create the branch to the given revision if that revision doesn't exist.
        """
        command = f"git checkout {revision}"
        try:
            result = subprocess.run(
                command.split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            )
            logger.warning(f"Checked out {revision} from {self.current_branch}.")
            logger.warning(result.stdout)
        except subprocess.CalledProcessError as exc:
            if not create_branch_ok:
                raise EnvironmentError(exc.stderr)
            else:
                command = f"git checkout -b {revision}"
                try:
                    result = subprocess.run(
                        command.split(),
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        check=True,
                        encoding="utf-8",
                        cwd=self.local_dir,
                    )
                    logger.warning(
                        f"Revision `{revision}` does not exist. Created and checked out branch `{revision}`."
                    )
                    logger.warning(result.stdout)
                except subprocess.CalledProcessError as exc:
                    raise EnvironmentError(exc.stderr)

    def tag_exists(self, tag_name: str, remote: Optional[str] = None) -> bool:
        """
        Check if a tag exists or not
        """
        if remote:
            try:
                result = subprocess.run(
                    ["git", "ls-remote", "origin", f"refs/tags/{tag_name}"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                ).stdout.strip()
            except subprocess.CalledProcessError as exc:
                raise EnvironmentError(exc.stderr)

            return len(result) != 0
        else:
            try:
                git_tags = subprocess.run(
                    ["git", "tag"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                ).stdout.strip()
            except subprocess.CalledProcessError as exc:
                raise EnvironmentError(exc.stderr)

            git_tags = git_tags.split("\n")
            return tag_name in git_tags

    def delete_tag(self, tag_name: str, remote: Optional[str] = None) -> bool:
        """
        Delete a tag, both local and remote, if it exists

        Return True if deleted.  Returns False if the tag didn't exist
        If remote is None, will just be updated locally
        """
        delete_locally = True
        delete_remotely = True

        if not self.tag_exists(tag_name):
            delete_locally = False

        if not self.tag_exists(tag_name, remote=remote):
            delete_remotely = False

        if delete_locally:
            try:
                subprocess.run(
                    ["git", "tag", "-d", tag_name],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                ).stdout.strip()
            except subprocess.CalledProcessError as exc:
                raise EnvironmentError(exc.stderr)

        if remote and delete_remotely:
            try:
                subprocess.run(
                    ["git", "push", remote, "--delete", tag_name],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                ).stdout.strip()
            except subprocess.CalledProcessError as exc:
                raise EnvironmentError(exc.stderr)

        return True

    def add_tag(self, tag_name: str, message: str = None, remote: Optional[str] = None):
        """
        Add a tag at the current head and push it

        If remote is None, will just be updated locally

        If no message is provided, the tag will be lightweight.
        if a message is provided, the tag will be annotated.
        """
        if message:
            tag_args = ["git", "tag", "-a", tag_name, "-m", message]
        else:
            tag_args = ["git", "tag", tag_name]
        try:
            subprocess.run(
                tag_args,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

        if remote:
            try:
                subprocess.run(
                    ["git", "push", remote, tag_name],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                    encoding="utf-8",
                    cwd=self.local_dir,
                ).stdout.strip()
            except subprocess.CalledProcessError as exc:
                raise EnvironmentError(exc.stderr)

    def is_repo_clean(self) -> bool:
        """
        Return whether or not the git status is clean or not
        """
        try:
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=self.local_dir,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(exc.stderr)

        return len(git_status) == 0

    def push_to_hub(
        self,
        commit_message: Optional[str] = "commit files to HF hub",
        blocking: Optional[bool] = True,
        clean_ok: Optional[bool] = True,
        auto_lfs_prune: Optional[bool] = False,
    ) -> Optional[str]:
        """
        Helper to add, commit, and push files to remote repository on the HuggingFace Hub.
        Will automatically track large files (>10MB).

        Args:
            commit_message (`str`):
                Message to use for the commit.
            blocking (`bool`, `optional`, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            clean_ok (`bool`, `optional`, defaults to `True`):
                If True, this function will return None if the repo is untouched.
                Default behavior is to fail because the git command fails.
            auto_lfs_prune (`bool`, defaults to `False`):
                Whether to automatically prune files once they have been pushed to the remote.
        """
        if clean_ok and self.is_repo_clean():
            logger.info("Repo currently clean.  Ignoring push_to_hub")
            return None
        self.git_add(auto_lfs_track=True)
        self.git_commit(commit_message)
        return self.git_push(
            upstream=f"origin {self.current_branch}",
            blocking=blocking,
            auto_lfs_prune=auto_lfs_prune,
        )

    @contextmanager
    def commit(
        self,
        commit_message: str,
        branch: Optional[str] = None,
        track_large_files: Optional[bool] = True,
        blocking: Optional[bool] = True,
        auto_lfs_prune: Optional[bool] = False,
    ):
        """
        Context manager utility to handle committing to a repository. This automatically tracks large files (>10Mb)
        with git-lfs. Set the `track_large_files` argument to `False` if you wish to ignore that behavior.

        Args:
            commit_message (`str`):
                Message to use for the commit.
            branch (`str`, `optional`):
                The branch on which the commit will appear. This branch will be checked-out before any operation.
            track_large_files (`bool`, `optional`, defaults to `True`):
                Whether to automatically track large files or not. Will do so by default.
            blocking (`bool`, `optional`, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            auto_lfs_prune (`bool`, defaults to `True`):
                Whether to automatically prune files once they have been pushed to the remote.

        Examples:

            >>> with Repository("text-files", clone_from="<user>/text-files", use_auth_token=True).commit("My first file :)"):
            ...     with open("file.txt", "w+") as f:
            ...         f.write(json.dumps({"hey": 8}))

            >>> import torch
            >>> model = torch.nn.Transformer()
            >>> with Repository("torch-model", clone_from="<user>/torch-model", use_auth_token=True).commit("My cool model :)"):
            ...     torch.save(model.state_dict(), "model.pt")

        """

        files_to_stage = files_to_be_staged(".", folder=self.local_dir)

        if len(files_to_stage):
            if len(files_to_stage) > 5:
                files_to_stage = str(files_to_stage[:5])[:-1] + ", ...]"

            logger.error(
                f"There exists some updated files in the local repository that are not committed: {files_to_stage}. "
                "This may lead to errors if checking out a branch. "
                "These files and their modifications will be added to the current commit."
            )

        if branch is not None:
            self.git_checkout(branch, create_branch_ok=True)

        if is_tracked_upstream(self.local_dir):
            logger.warning("Pulling changes ...")
            self.git_pull(rebase=True)
        else:
            logger.warning(
                f"The current branch has no upstream branch. Will push to 'origin {self.current_branch}'"
            )

        current_working_directory = os.getcwd()
        os.chdir(os.path.join(current_working_directory, self.local_dir))

        try:
            yield self
        finally:
            self.git_add(auto_lfs_track=track_large_files)

            try:
                self.git_commit(commit_message)
            except OSError as e:
                # If no changes are detected, there is nothing to commit.
                if "nothing to commit" not in str(e):
                    raise e

            try:
                self.git_push(
                    upstream=f"origin {self.current_branch}",
                    blocking=blocking,
                    auto_lfs_prune=auto_lfs_prune,
                )
            except OSError as e:
                # If no changes are detected, there is nothing to commit.
                if "could not read Username" in str(e):
                    raise OSError(
                        "Couldn't authenticate user for push. Did you set `use_auth_token` to `True`?"
                    ) from e
                else:
                    raise e

            os.chdir(current_working_directory)

    def repocard_metadata_load(self) -> Optional[Dict]:
        filepath = os.path.join(self.local_dir, REPOCARD_NAME)
        if os.path.isfile(filepath):
            return metadata_load(filepath)

    def repocard_metadata_save(self, data: Dict) -> None:
        return metadata_save(os.path.join(self.local_dir, REPOCARD_NAME), data)

    @property
    def commands_failed(self):
        return [c for c in self.command_queue if c.status > 0]

    @property
    def commands_in_progress(self):
        return [c for c in self.command_queue if not c.is_done]

    def wait_for_commands(self):
        index = 0
        for command_failed in self.commands_failed:
            logger.error(
                f"The {command_failed.title} command with PID {command_failed._process.pid} failed."
            )
            logger.error(command_failed.stderr)

        while self.commands_in_progress:
            if index % 10 == 0:
                logger.error(
                    f"Waiting for the following commands to finish before shutting down: {self.commands_in_progress}."
                )

            index += 1

            time.sleep(1)
