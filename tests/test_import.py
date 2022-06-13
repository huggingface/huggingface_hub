import sys


def test_import_huggingface_hub_doesnt_import_tensorfow():
    import huggingface_hub  # noqa

    assert "tensorflow" not in sys.modules
