from huggingface_hub import get_token


def test_no_token_in_staging_environment():
    """Make sure no token is set in test environment."""
    assert get_token() is None

from pathlib import Path
import yaml


def test_python_tests_workflow_case_statements():
    """Ensure python-tests.yml handles all matrix test_names and has fallback arms."""
    workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "python-tests.yml"
    with open(workflow_path) as f:
        data = yaml.safe_load(f)

    steps = data["jobs"]["build-ubuntu"]["steps"]
    run_step = next(s for s in steps if s.get("name") == "Run tests")
    install_step = next(s for s in steps if "uv pip install" in s.get("run", "") and "case" in s.get("run", ""))

    assert "*)" in install_step["run"], "Install step case is missing default fallback arm (*)"
    assert "*)" in run_step["run"], "Run tests step case is missing default fallback arm (*)"
