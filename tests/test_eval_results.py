import pytest

from huggingface_hub import EvalResultEntry, eval_result_entries_to_yaml, parse_eval_result_entries


def test_eval_result_entry_minimal():
    entry = EvalResultEntry(dataset_id="openai/gsm8k", value=86.2, task_id="main")
    assert entry.dataset_id == "openai/gsm8k"
    assert entry.value == 86.2
    assert entry.task_id == "main"


def test_eval_result_entry_source_requires_url():
    with pytest.raises(ValueError):
        EvalResultEntry(dataset_id="test", value=1.0, task_id="main", source_name="Test")
    with pytest.raises(ValueError):
        EvalResultEntry(dataset_id="test", value=1.0, task_id="main", source_org="test-org")


def test_eval_result_entries_to_yaml():
    entries = [EvalResultEntry(dataset_id="openai/gsm8k", value=86.2, task_id="main")]
    result = eval_result_entries_to_yaml(entries)
    assert result == [{"dataset": {"id": "openai/gsm8k", "task_id": "main"}, "value": 86.2}]


def test_parse_eval_result_entries():
    data = [{"dataset": {"id": "openai/gsm8k", "task_id": "main"}, "value": 86.2}]
    entries = parse_eval_result_entries(data)
    assert len(entries) == 1
    assert entries[0].dataset_id == "openai/gsm8k"
    assert entries[0].value == 86.2
    assert entries[0].task_id == "main"


def test_parse_eval_result_entries_api_format():
    """Test parsing the API response format (with data wrapper)."""
    data = [
        {
            "filename": ".eval_results/gsm8k.yaml",
            "verified": False,
            "data": {
                "dataset": {"id": "openai/gsm8k", "task_id": "main"},
                "value": 86.2,
                "date": "2024-04-22",
                "source": {"url": "https://hf.co/papers/2404.14219", "name": "Phi-3 Technical Report"},
            },
            "pullRequest": 44,
        }
    ]
    entries = parse_eval_result_entries(data)
    assert len(entries) == 1
    assert entries[0].dataset_id == "openai/gsm8k"
    assert entries[0].task_id == "main"
    assert entries[0].value == 86.2
    assert entries[0].date == "2024-04-22"
    assert entries[0].source_url == "https://hf.co/papers/2404.14219"
    assert entries[0].source_name == "Phi-3 Technical Report"
