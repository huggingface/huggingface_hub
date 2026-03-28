<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Mixins & serialization methods

## Mixins

The `huggingface_hub` library offers a range of mixins that can be used as a parent class for your objects, in order to
provide simple uploading and downloading functions. Check out our [integration guide](../guides/integrations) to learn
how to integrate any ML framework with the Hub.

### Generic

[[autodoc]] ModelHubMixin
    - all
    - _save_pretrained
    - _from_pretrained

### PyTorch

[[autodoc]] PyTorchModelHubMixin

### Fastai

[[autodoc]] from_pretrained_fastai

[[autodoc]] push_to_hub_fastai
