import time

import pytest
from sentence_transformers import SentenceTransformer, util

from huggingface_hub import model_info

from ..utils import production_endpoint


@pytest.fixture(scope="module")
def multi_qa_model() -> SentenceTransformer:
    with production_endpoint():
        return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def test_from_pretrained(multi_qa_model: SentenceTransformer) -> None:
    # Example taken from https://www.sbert.net/docs/hugging_face.html#using-hugging-face-models.
    query_embedding = multi_qa_model.encode("How big is London")
    passage_embedding = multi_qa_model.encode(
        [
            "London has 9,787,426 inhabitants at the 2011 census",
            "London is known for its financial district",
        ]
    )
    print("Similarity:", util.dot_score(query_embedding, passage_embedding))


def test_push_to_hub(multi_qa_model: SentenceTransformer, repo_name: str, user: str, cleanup_repo: None) -> None:
    multi_qa_model.save_to_hub(repo_name, organization=user)

    # Sleep to ensure that model_info isn't called too soon
    time.sleep(1)

    # Check model has been pushed properly
    model_id = f"{user}/{repo_name}"
    assert model_info(model_id).library_name == "sentence-transformers"
