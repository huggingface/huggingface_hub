from sentence_transformers import SentenceTransformer, util

import pytest
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


@pytest.mark.xfail(reason="Production endpoint is hardcoded in sentence_transformers when pushing to Hub.")
def test_push_to_hub(
    multi_qa_model: SentenceTransformer, repo_name: str, cleanup_repo: None
) -> None:
    multi_qa_model.save_to_hub(repo_name)
