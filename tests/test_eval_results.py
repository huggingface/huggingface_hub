import pytest

from huggingface_hub import EvalResultEntry, eval_result_entries_to_yaml, parse_eval_result_entries


def test_eval_result_entry_minimal():
    entry = EvalResultEntry(dataset_id="cais/hle", value=20.90)
    assert entry.dataset_id == "cais/hle"
    assert entry.value == 20.90


def test_eval_result_entry_source_requires_url():
    with pytest.raises(ValueError):
        EvalResultEntry(dataset_id="test", value=1.0, source_name="Test")


def test_eval_result_entries_to_yaml():
    entries = [EvalResultEntry(dataset_id="cais/hle", value=20.90, task_id="default")]
    result = eval_result_entries_to_yaml(entries)
    assert result == [{"dataset": {"id": "cais/hle", "task_id": "default"}, "value": 20.90}]


def test_parse_eval_result_entries_new_format():
    data = [{"dataset": {"id": "cais/hle", "task_id": "default"}, "value": 20.90}]
    entries = parse_eval_result_entries(data)
    assert len(entries) == 1
    assert entries[0].dataset_id == "cais/hle"
    assert entries[0].value == 20.90


def test_parse_eval_result_entries_legacy_format():
    data = [
        {
            "dataset": {"type": "Idavidrein/gpqa", "config": "gpqa_diamond"},
            "metrics": [{"type": "accuracy", "value": 0.412}],
        }
    ]
    entries = parse_eval_result_entries(data)
    assert len(entries) == 1
    assert entries[0].dataset_id == "Idavidrein/gpqa"
    assert entries[0].value == 0.412
