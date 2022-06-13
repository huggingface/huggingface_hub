import sys
import unittest

from huggingface_hub.file_download import is_tf_available


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow, graphviz and pydot.

    These tests are skipped when TensorFlow, graphviz and pydot are installed.

    """
    if not is_tf_available():
        return unittest.skip("test requires Tensorflow")(test_case)
    else:
        return test_case


@require_tf
def test_import_huggingface_hub_doesnt_import_tensorfow():
    # Not necessary since huggingface_hub is already imported at the top of this file,
    # but let's keep it here anyway just in case
    import huggingface_hub  # noqa

    assert "tensorflow" not in sys.modules
