import os
import subprocess
from typing import Dict, List


MODEL_CARD_TEMPLATE = """
---
tags:
- sagemaker
datasets:
- {dataset}
---
## {model_id} Trained from SageMaker HuggingFace extension.

#### Hyperparameters
```json
{hyperparameters}
```

#### Eval
| key | value |
| --- | ----- |
{eval_table}
"""


class HfRepository:
    """Git-based system for HuggingFace Hub repositories"""

    def __init__(
        self,
        repo_url: str,
        huggingface_token: str,
        model_dir: str,
    ):
        self.repo_url = self.add_token_to_repository_url(repo_url, huggingface_token)

        self.model_dir = model_dir

    def init_new_repository(
        self,
        large_file_tracking_list=[],
        user="sagemaker",
        email="sagemaker@huggingface.co",
    ):
        """Initializes a new hf hub repository"""

        self.email = email
        self.user = user
        self.large_file_tracking_list = [
            "*.bin.*",
            "*.lfs.*",
            "*.bin",
            "*.h5",
            "*.tflite",
            "*.tar.gz",
            "*.ot",
            "*.onnx",
            "*.pt",
        ] + large_file_tracking_list

        self.install_lfs_in_repo()

        self.clone_remote_repository_to_local_dir()

        self.add_tracking_for_large_files()

        # self.config_git_username_and_email()

    def install_lfs_in_repo(self):
        """installs git lfs for current shell user"""
        subprocess.run("git lfs install".split(), check=True)

    def config_git_username_and_email(self):
        """sets git user and email for commiting to repository"""
        subprocess.run(f"git config --global user.email {self.email}".split(), check=True)
        subprocess.run(f"git config --global user.name {self.user}".split(), check=True)

    def clone_remote_repository_to_local_dir(self):
        """clones repository from HuggingFace Hub into model directory if directory exists init and add remote repo"""
        # TODO: refactor exception handling to avoidtry:
        #   try:
        #     output = subprocess.check_output(['git', 'commit', '-m', 'commit model from SageMaker'])
        #   except subprocess.CalledProcessError as e:
        #       print(e.output.decode())
        try:
            subprocess.run(f"git clone {self.repo_url} {self.model_dir}".split(), check=True)
        except Exception:
            subprocess.run(f"git init .".split(), check=True, cwd=self.model_dir)
            subprocess.run(f"git remote add origin {self.repo_url}".split(), check=True, cwd=self.model_dir)
            subprocess.run(f"git checkout -b main".split(), check=True, cwd=self.model_dir)

    def add_token_to_repository_url(self, repo_url: str, huggingface_token: str) -> str:
        """replaces default repository url and adds huggingface token to so git push is possible"""
        return repo_url.replace("https://", f"https://user:{huggingface_token}@")

    def add_tracking_for_large_files(self):
        """adds all large files to .gitattributes for tracking"""
        for file_ending in self.large_file_tracking_list:
            subprocess.run(f"git lfs track {file_ending}".split(), check=True, cwd=self.model_dir)

    def create_model_card(self, dataset: str, model_id: str, hyperparameters: Dict, eval_results: List):
        """creates a model card from an existing template needs datasets, model_id, hyperparameters and eval_results"""
        model_card = MODEL_CARD_TEMPLATE.format(
            dataset=dataset,
            model_id=model_id,
            hyperparameters=hyperparameters,
            eval_table="\n".join(f"| {k} | {v} |" for k, v in eval_results.items()),
        )
        with open(os.path.join(self.model_dir, "README.md"), "w") as f:
            f.write(model_card)

        # self.commit_files_and_push_to_hub("added model card")

    def commit_files(self, commit_message="commit model from SageMaker"):
        """adds and commits files from current directory"""
        subprocess.run(f"git add .".split(), check=True, cwd=self.model_dir)
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=self.model_dir)

    def push_to_hub(self):
        """pushes commited files to HuggingFace Hub"""
        # TODO: refactor exception handling to avoidtry:
        #   try:
        #     output = subprocess.check_output(['git', 'commit', '-m', 'commit model from SageMaker'])
        #   except subprocess.CalledProcessError as e:
        #       print(e.output.decode())
        try:
            subprocess.run(f"git push".split(), check=True, cwd=self.model_dir)
        except Exception:
            subprocess.run(f"git fetch".split(), check=True, cwd=self.model_dir)
            subprocess.run(f"git branch --set-upstream-to=origin/main main".split(), check=True, cwd=self.model_dir)
            subprocess.run(f"git pull --allow-unrelated-histories".split(), check=True, cwd=self.model_dir)
            subprocess.run(f"git push".split(), check=True, cwd=self.model_dir)

    def commit_files_and_push_to_hub(self, commit_message="commit model from SageMaker"):
        """combines commit_files and push_to_hub"""
        self.commit_files(commit_message)
        self.push_to_hub()
