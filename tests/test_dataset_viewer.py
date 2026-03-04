import json

from huggingface_hub import HfApi
from huggingface_hub._dataset_viewer import execute_raw_sql_query
from huggingface_hub.constants import _HF_DEFAULT_ENDPOINT


def test_execute_raw_sql_query_with_duckdb_on_public_parquet() -> None:
    entries = HfApi(endpoint=_HF_DEFAULT_ENDPOINT).list_dataset_parquet_files(
        repo_id="nvidia/Llama-Nemotron-Post-Training-Dataset",
        token=False,
    )
    assert len(entries) > 0
    parquet_url = entries[0].url
    query = f"SELECT * FROM read_parquet('{parquet_url}') LIMIT 1"

    result = execute_raw_sql_query(sql_query=query, token=False, output_format="json")

    if result.raw_json is not None:
        payload = json.loads(result.raw_json)
        assert len(payload) == 1
    else:
        assert len(result.columns) > 0
        assert len(result.rows) == 1
