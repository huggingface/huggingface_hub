import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from huggingface_hub import snapshot_download, upload_file


DATASET_ID = "tiny-agents/tiny-agents"

CONFIG_FILE = "agent.json"
PROMPT_FILE = "PROMPT.md"
README_FILE = "README.md"

folder = Path(snapshot_download(repo_id=DATASET_ID, repo_type="dataset"))


def load_agent(agent_path: Path):
    """Load agent configuration from a given path."""
    config_path = agent_path / CONFIG_FILE
    prompt_path = agent_path / PROMPT_FILE
    readme_path = agent_path / README_FILE

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found in {agent_path}")

    config = json.dumps(json.loads(config_path.read_text()), indent=2)
    prompt = prompt_path.read_text() if prompt_path.exists() else None
    readme = readme_path.read_text() if readme_path.exists() else None
    return {
        "name": f"{agent_path.parent.name}/{agent_path.name}",
        "config": config,
        "readme": readme,
        "prompt": prompt,
    }


AGENTS = []
for namespace in folder.iterdir():
    if namespace.is_dir():
        for agent in namespace.iterdir():
            if agent.is_dir():
                try:
                    AGENTS.append(load_agent(agent))
                except Exception as e:
                    print(f"Skipping {agent.name}: {e}")
AGENTS.sort(key=lambda x: x["name"])

# Export as parquet file
with TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / "data.parquet"
    pd.DataFrame(AGENTS).to_parquet(tmp_path, index=False)
    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="data.parquet",
        repo_id=DATASET_ID,
        repo_type="dataset",
        commit_message="Update summary for dataset-viewer",
    )
