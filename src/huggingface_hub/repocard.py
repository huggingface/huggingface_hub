import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jinja2
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
    CardData,
    EvalResult,
    eval_results_to_model_index,
    model_index_to_eval_results,
)

from .constants import REPOCARD_NAME
from .utils.logging import get_logger


# exact same regex as in the Hub server. Please keep in sync.
TEMPLATE_MODELCARD_PATH = Path(__file__).parent / "templates" / "modelcard_template.md"
REGEX_YAML_BLOCK = re.compile(r"---[\n\r]+([\S\s]*?)[\n\r]+---[\n\r]")

logger = get_logger(__name__)


class RepoCard:
    def __init__(self, content: str):
        """Initialize a RepoCard from string content. The content should be a
        Markdown file with a YAML block at the beginning and a Markdown body.

        Args:
            content (`str`): The content of the Markdown file.

        Raises:
            ValueError: When the content of the repo card metadata is not found.
            ValueError: When the content of the repo card metadata is not a dictionary.
        """
        self.content = content
        match = REGEX_YAML_BLOCK.search(content)
        if match:
            # Metadata found in the YAML block
            yaml_block = match.group(1)
            self.text = content[match.end() :]
            data_dict = yaml.safe_load(yaml_block)

            # The YAML block's data should be a dictionary
            if not isinstance(data_dict, dict):
                raise ValueError("repo card metadata block should be a dict")
        else:
            # Model card without metadata... create empty metadata
            logger.warning(
                "Repo card metadata block was not found. Setting CardData to empty."
            )
            data_dict = {}
            self.text = content

        model_index = data_dict.pop("model-index", None)
        if model_index:
            try:
                model_name, eval_results = model_index_to_eval_results(model_index)
                data_dict["model_name"] = model_name
                data_dict["eval_results"] = eval_results
            except KeyError:
                logger.warning(
                    "Invalid model-index. Not loading eval results into CardData."
                )

        self.data = CardData(**data_dict)

    def __str__(self):
        return f"---\n{self.data.to_yaml()}\n---\n{self.text}"

    def save(self, filepath: Union[Path, str]):
        r"""Save a RepoCard to a file.

        Args:
            filepath (`Union[Path, str]`): Filepath to the markdown file to save.

        Example:
            >>> from huggingface_hub import RepoCard
            >>> card = RepoCard("---\nlanguage: en\n---\n# This is a test repo card")
            >>> card.save("/tmp/test.md")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(str(self))

    @classmethod
    def load(cls, repo_id_or_path: Union[str, Path], repo_type=None, token=None):
        """Initialize a RepoCard from a Hugging Face Hub repo's README.md or a local filepath.

        Args:
            repo_id_or_path (`Union[str, Path]`):
                The repo ID associated with a Hugging Face Hub repo or a local filepath.
            repo_type (`str`, *optional*):
                The type of Hugging Face repo to push to. Defaults to None, which will use
                use "model". Other options are "dataset" and "space".
            token (`str`, *optional*):
                Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to
                the stored token.

        Returns:
            `huggingface_hub.RepoCard`: The RepoCard (or subclass) initialized from the repo's
                README.md file or filepath.

        Example:
            >>> from huggingface_hub import RepoCard
            >>> card = RepoCard.load("nateraw/food")
            >>> assert card.data.tags == ["generated_from_trainer", "image-classification", "pytorch"]
        """

        if Path(repo_id_or_path).exists():
            card_path = Path(repo_id_or_path)
        else:
            card_path = hf_hub_download(
                repo_id_or_path,
                REPOCARD_NAME,
                repo_type=repo_type,
                use_auth_token=token,
            )

        return cls(Path(card_path).read_text())

    def validate(self, repo_type="model"):
        """Validates card against Hugging Face Hub's model card validation logic.
        Using this function requires access to the internet, so it is only called
        internally by `huggingface_hub.ModelCard.push_to_hub`.

        Args:
            repo_type (`str`, *optional*):
                The type of Hugging Face repo to push to. Defaults to None, which will use
                use "model". Other options are "dataset" and "space".
        """
        if repo_type is None:
            repo_type = "model"

        # TODO - compare against repo types constant in huggingface_hub if we move this object there.
        if repo_type not in ["model", "space", "dataset"]:
            raise RuntimeError(
                "Provided repo_type '{repo_type}' should be one of ['model', 'space',"
                " 'dataset']."
            )

        body = {
            "repoType": repo_type,
            "content": str(self),
        }
        headers = {"Accept": "text/plain"}

        try:
            r = requests.post(
                "https://huggingface.co/api/validate-yaml", body, headers=headers
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if r.status_code == 400:
                raise RuntimeError(r.text)
            else:
                raise exc

    def push_to_hub(
        self,
        repo_id,
        token=None,
        repo_type=None,
        commit_message=None,
        commit_description=None,
        revision=None,
        create_pr=None,
    ):
        """Push a RepoCard to a Hugging Face Hub repo.

        Args:
            repo_id (`str`):
                The repo ID of the Hugging Face Hub repo to push to. Example: "nateraw/food".
            token (`str`, *optional*):
                Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to
                the stored token.
            repo_type (`str`, *optional*):
                The type of Hugging Face repo to push to. Defaults to None, which will use
                use "model". Other options are "dataset" and "space".
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit
            commit_description (`str`, *optional*)
                The description of the generated commit
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the
                `"main"` branch.
            create_pr (`bool`, *optional*):
                Whether or not to create a Pull Request with this commit. Defaults to `False`.
        Returns:
            `str`: URL of the commit which updated the card metadata.
        """

        # TODO - Remove this if we decide updating the name is no bueno.
        # This breaks unittests on updating metadata that includes name that != repo name

        # repo_name = repo_id.split("/")[-1]

        # if self.data.model_name and self.data.model_name != repo_name:
        #     logger.warning(
        #         f"Set model name {self.data.model_name} in CardData does not match "
        #         f"repo name {repo_name}. Updating model name to match repo name."
        #     )
        #     self.data.model_name = repo_name

        # Validate card before pushing to hub
        self.validate(repo_type=repo_type)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / REPOCARD_NAME
            tmp_path.write_text(str(self))
            url = upload_file(
                path_or_fileobj=str(tmp_path),
                path_in_repo=REPOCARD_NAME,
                repo_id=repo_id,
                token=token,
                repo_type=repo_type,
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
                revision=revision,
            )
        return url


class ModelCard(RepoCard):
    @classmethod
    def from_template(
        cls,
        card_data: CardData,
        template_path: Optional[str] = TEMPLATE_MODELCARD_PATH,
        **template_kwargs,
    ):
        """Initialize a ModelCard from a template. By default, it uses the default template.

        Templates are Jinja2 templates that can be customized by passing keyword arguments.

        Args:
            card_data (`huggingface_hub.CardData`):
                A huggingface_hub.CardData instance containing the metadata you want to include in the YAML
                header of the model card on the Hugging Face Hub.
            template_path (`str`, *optional*):
                A path to a markdown file with optional Jinja template variables that can be filled
                in with `template_kwargs`. Defaults to the default template. # TODO - add link here

        Returns:
            `huggingface_hub.ModelCard`: A ModelCard instance with the specified card data and content from the
            template.

        Example:
            >>> from huggingface_hub import ModelCard, CardData, EvalResult

            >>> # Using the Default Template
            >>> card_data = CardData(
            ...     language='en',
            ...     license='mit',
            ...     library_name='timm',
            ...     tags=['image-classification', 'resnet'],
            ...     datasets='beans',
            ...     metrics=['accuracy'],
            ... )
            >>> card = ModelCard.from_template(
            ...     card_data,
            ...     model_description='This model does x + y...'
            ... )

            >>> # Including Evaluation Results
            >>> card_data = CardData(
            ...     language='en',
            ...     tags=['image-classification', 'resnet'],
            ...     eval_results=[
            ...         EvalResult(
            ...             task_type='image-classification',
            ...             dataset_type='beans',
            ...             dataset_name='Beans',
            ...             metric_type='accuracy',
            ...             metric_value=0.9,
            ...         ),
            ...     ],
            ...     model_name='my-cool-model',
            ... )
            >>> card = ModelCard.from_template(card_data)

            >>> # Using a Custom Template
            >>> card_data = CardData(
            ...     language='en',
            ...     tags=['image-classification', 'resnet']
            ... )
            >>> card = ModelCard.from_template(
            ...     card_data=card_data,
            ...     template_path='./src/huggingface_hub/modelcard_template.md',
            ...     custom_template_var='custom value',  # will be replaced in template if it exists
            ... )

        """
        content = jinja2.Template(Path(template_path).read_text()).render(
            card_data=card_data.to_yaml(), **template_kwargs
        )
        return cls(content)


def metadata_load(local_path: Union[str, Path]) -> Optional[Dict]:
    content = Path(local_path).read_text()
    match = REGEX_YAML_BLOCK.search(content)
    if match:
        yaml_block = match.group(1)
        data = yaml.safe_load(yaml_block)
        if isinstance(data, dict):
            return data
        else:
            raise ValueError("repo card metadata block should be a dict")
    else:
        return None


def metadata_save(local_path: Union[str, Path], data: Dict) -> None:
    """
    Save the metadata dict in the upper YAML part Trying to preserve newlines as
    in the existing file. Docs about open() with newline="" parameter:
    https://docs.python.org/3/library/functions.html?highlight=open#open Does
    not work with "^M" linebreaks, which are replaced by \n
    """
    line_break = "\n"
    content = ""
    # try to detect existing newline character
    if os.path.exists(local_path):
        with open(local_path, "r", newline="") as readme:
            if type(readme.newlines) is tuple:
                line_break = readme.newlines[0]
            if type(readme.newlines) is str:
                line_break = readme.newlines
            content = readme.read()

    # creates a new file if it not
    with open(local_path, "w", newline="") as readme:
        data_yaml = yaml.dump(data, sort_keys=False, line_break=line_break)
        # sort_keys: keep dict order
        match = REGEX_YAML_BLOCK.search(content)
        if match:
            output = (
                content[: match.start()]
                + f"---{line_break}{data_yaml}---{line_break}"
                + content[match.end() :]
            )
        else:
            output = f"---{line_break}{data_yaml}---{line_break}{content}"

        readme.write(output)
        readme.close()


def metadata_eval_result(
    model_pretty_name: str,
    task_pretty_name: str,
    task_id: str,
    metrics_pretty_name: str,
    metrics_id: str,
    metrics_value: Any,
    dataset_pretty_name: str,
    dataset_id: str,
    metrics_config: Optional[str] = None,
    metrics_verified: Optional[bool] = False,
    dataset_config: Optional[str] = None,
    dataset_split: Optional[str] = None,
    dataset_revision: Optional[str] = None,
) -> Dict:
    """
    Creates a metadata dict with the result from a model evaluated on a dataset.

    Args:
        model_pretty_name (`str`):
            The name of the model in natural language.
        task_pretty_name (`str`):
            The name of a task in natural language.
        task_id (`str`):
            Example: automatic-speech-recognition. A task id.
        metrics_pretty_name (`str`):
            A name for the metric in natural language. Example: Test WER.
        metrics_id (`str`):
            Example: wer. A metric id from https://hf.co/metrics.
        metrics_value (`Any`):
            The value from the metric. Example: 20.0 or "20.0 ± 1.2".
        dataset_pretty_name (`str`):
            The name of the dataset in natural language.
        dataset_id (`str`):
            Example: common_voice. A dataset id from https://hf.co/datasets.
        metrics_config (`str`, *optional*):
            The name of the metric configuration used in `load_metric()`.
            Example: bleurt-large-512 in `load_metric("bleurt", "bleurt-large-512")`.
        metrics_verified (`bool`, *optional*, defaults to `False`):
            If true, indicates that evaluation was generated by Hugging Face (vs. self-reported).
            If a user tries to push self-reported metric results with verified=True, the push
            will be rejected.
        dataset_config (`str`, *optional*):
            Example: fr. The name of the dataset configuration used in `load_dataset()`.
        dataset_split (`str`, *optional*):
            Example: test. The name of the dataset split used in `load_dataset()`.
        dataset_revision (`str`, *optional*):
            Example: 5503434ddd753f426f4b38109466949a1217c2bb. The name of the dataset dataset revision
            used in `load_dataset()`.

    Returns:
        `dict`: a metadata dict with the result from a model evaluated on a dataset.

    Example:
    >>> from huggingface_hub import metadata_eval_result
    >>> metadata_eval_result(
    ...         model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
    ...         task_pretty_name="Text Classification",
    ...         task_id="text-classification",
    ...         metrics_pretty_name="Accuracy",
    ...         metrics_id="accuracy",
    ...         metrics_value=0.2662102282047272,
    ...         dataset_pretty_name="ReactionJPEG",
    ...         dataset_id="julien-c/reactionjpeg",
    ...         dataset_config="default",
    ...         dataset_split="test",
    ...     )
    {
        "model-index": [
            {
                "name": "RoBERTa fine-tuned on ReactionGIF",
                "results": [
                    {
                        "task": {
                            "type": "text-classification",
                            "name": "Text Classification",
                        },
                        "dataset": {
                            "name": "ReactionJPEG",
                            "type": "julien-c/reactionjpeg",
                            "config": "default",
                            "split": "test",
                        },
                        "metrics": [
                            {
                                "type": "accuracy",
                                "value": 0.2662102282047272,
                                "name": "Accuracy",
                                "verified": False,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    """
    return {
        "model-index": eval_results_to_model_index(
            model_name=model_pretty_name,
            eval_results=[
                EvalResult(
                    task_name=task_pretty_name,
                    task_type=task_id,
                    metric_name=metrics_pretty_name,
                    metric_type=metrics_id,
                    metric_value=metrics_value,
                    dataset_name=dataset_pretty_name,
                    dataset_type=dataset_id,
                    metric_config=metrics_config,
                    verified=metrics_verified,
                    dataset_config=dataset_config,
                    dataset_split=dataset_split,
                    dataset_revision=dataset_revision,
                )
            ],
        )
    }


def metadata_update(
    repo_id: str,
    metadata: Dict,
    *,
    repo_type: Optional[str] = None,
    overwrite: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    commit_description: Optional[str] = None,
    revision: Optional[str] = None,
    create_pr: bool = False,
) -> str:
    """
    Updates the metadata in the README.md of a repository on the Hugging Face Hub.

    Example:
    >>> from huggingface_hub import metadata_update
    >>> metadata = {'model-index': [{'name': 'RoBERTa fine-tuned on ReactionGIF',
    ...             'results': [{'dataset': {'name': 'ReactionGIF',
    ...                                      'type': 'julien-c/reactiongif'},
    ...                           'metrics': [{'name': 'Recall',
    ...                                        'type': 'recall',
    ...                                        'value': 0.7762102282047272}],
    ...                          'task': {'name': 'Text Classification',
    ...                                   'type': 'text-classification'}}]}]}
    >>> update_metdata("julien-c/reactiongif-roberta", metadata)

    Args:
        repo_id (`str`):
            The name of the repository.
        metadata (`dict`):
            A dictionary containing the metadata to be updated.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if updating to a dataset or space,
            `None` or `"model"` if updating to a model. Default is `None`.
        overwrite (`bool`, *optional*, defaults to `False`):
            If set to `True` an existing field can be overwritten, otherwise
            attempting to overwrite an existing field will cause an error.
        token (`str`, *optional*):
            The Hugging Face authentication token.
        commit_message (`str`, *optional*):
            The summary / title / first line of the generated commit. Defaults to
            `f"Update metdata with huggingface_hub"`
        commit_description (`str` *optional*)
            The description of the generated commit
        revision (`str`, *optional*):
            The git revision to commit from. Defaults to the head of the
            `"main"` branch.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `revision` with that commit.
            Defaults to `False`.
    Returns:
        `str`: URL of the commit which updated the card metadata.
    """
    commit_message = (
        commit_message
        if commit_message is not None
        else "Update metadata with huggingface_hub"
    )

    card = ModelCard.load(repo_id, token=token)

    for key, value in metadata.items():
        if key == "model-index":
            model_name, new_results = model_index_to_eval_results(value)
            if card.data.eval_results is None:
                card.data.eval_results = new_results
                card.data.model_name = model_name
            else:
                existing_results = card.data.eval_results

                for new_result in new_results:
                    result_found = False
                    for existing_result_index, existing_result in enumerate(
                        existing_results
                    ):
                        if all(
                            [
                                new_result.dataset_name == existing_result.dataset_name,
                                new_result.dataset_type == existing_result.dataset_type,
                                new_result.task_type == existing_result.task_type,
                                new_result.task_name == existing_result.task_name,
                                new_result.metric_name == existing_result.metric_name,
                                new_result.metric_type == existing_result.metric_type,
                            ]
                        ):
                            if (
                                new_result.metric_value != existing_result.metric_value
                                and not overwrite
                            ):
                                existing_str = (
                                    f"name: {new_result.metric_name}, type:"
                                    f" {new_result.metric_type}"
                                )
                                raise ValueError(
                                    "You passed a new value for the existing metric"
                                    f" '{existing_str}'. Set `overwrite=True` to"
                                    " overwrite existing metrics."
                                )
                            result_found = True
                            card.data.eval_results[existing_result_index] = new_result
                    if not result_found:
                        card.data.eval_results.append(new_result)
        else:
            if (
                hasattr(card.data, key)
                and getattr(card.data, key) is not None
                and not overwrite
                and getattr(card.data, key) != value
            ):
                raise ValueError(
                    f"""You passed a new value for the existing meta data field '{key}'. Set `overwrite=True` to overwrite existing metadata."""
                )
            else:
                setattr(card.data, key, value)

    return card.push_to_hub(
        repo_id,
        token=token,
        repo_type=repo_type,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
        revision=revision,
    )
