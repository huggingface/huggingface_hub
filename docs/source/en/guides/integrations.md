<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Integrate any ML framework with the Hub

The Hugging Face Hub makes hosting and sharing models with the community easy. It supports
[dozens of libraries](https://huggingface.co/docs/hub/models-libraries) in the Open Source ecosystem. We are always
working on expanding this support to push collaborative Machine Learning forward. The `huggingface_hub` library plays a
key role in this process, allowing any Python script to easily push and load files.

There are four main ways to integrate a library with the Hub:
1. **Push to Hub:** implement a method to upload a model to the Hub. This includes the model weights, as well as
   [the model card](https://huggingface.co/docs/huggingface_hub/how-to-model-cards) and any other relevant information
   or data necessary to run the model (for example, training logs). This method is often called `push_to_hub()`.
2. **Download from Hub:** implement a method to load a model from the Hub. The method should download the model
   configuration/weights and load the model. This method is often called `from_pretrained` or `load_from_hub()`.
3. **Inference API:** use our servers to run inference on models supported by your library for free.
4. **Widgets:** display a widget on the landing page of your models on the Hub. It allows users to quickly try a model
   from the browser.

In this guide, we will focus on the first two topics. We will present the two main approaches you can use to integrate
a library, with their advantages and drawbacks. Everything is summarized at the end of the guide to help you choose
between the two. Please keep in mind that these are only guidelines that you are free to adapt to you requirements.

If you are interested in Inference and Widgets, you can follow [this guide](https://huggingface.co/docs/hub/models-adding-libraries#set-up-the-inference-api).
In both cases, you can reach out to us if you are integrating a library with the Hub and want to be listed
[in our docs](https://huggingface.co/docs/hub/models-libraries).

## A flexible approach: helpers

The first approach to integrate a library to the Hub is to actually implement the `push_to_hub` and `from_pretrained`
methods by yourself. This gives you full flexibility on which files you need to upload/download and how to handle inputs
specific to your framework. You can refer to the two [upload files](./upload) and [download files](./download) guides
to learn more about how to do that. This is, for example how the FastAI integration is implemented (see [`push_to_hub_fastai`]
and [`from_pretrained_fastai`]).

Implementation can differ between libraries, but the workflow is often similar.

### from_pretrained

This is how a `from_pretrained` method usually look like:

```python
def from_pretrained(model_id: str) -> MyModelClass:
   # Download model from Hub
   cached_model = hf_hub_download(
      repo_id=repo_id,
      filename="model.pkl",
      library_name="fastai",
      library_version=get_fastai_version(),
   )

   # Load model
    return load_model(cached_model)
```

### push_to_hub

The `push_to_hub` method often requires a bit more complexity to handle repo creation, generate the model card and save weights.
A common approach is to save all of these files in a temporary folder, upload it and then delete it.

```python
def push_to_hub(model: MyModelClass, repo_name: str) -> None:
   api = HfApi()

   # Create repo if not existing yet and get the associated repo_id
   repo_id = api.create_repo(repo_name, exist_ok=True)

   # Save all files in a temporary directory and push them in a single commit
   with TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      # Save weights
      save_model(model, tmpdir / "model.safetensors")

      # Generate model card
      card = generate_model_card(model)
      (tmpdir / "README.md").write_text(card)

      # Save logs
      # Save figures
      # Save evaluation metrics
      # ...

      # Push to hub
      return api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
```

This is of course only an example. If you are interested in more complex manipulations (delete remote files, upload
weights on the fly, persist weights locally, etc.) please refer to the [upload files](./upload) guide.

### Limitations

While being flexible, this approach has some drawbacks, especially in terms of maintenance. Hugging Face users are often
used to additional features when working with `huggingface_hub`. For example, when loading files from the Hub, it is
common to offer parameters like:
- `token`: to download from a private repo
- `revision`: to download from a specific branch
- `cache_dir`: to cache files in a specific directory
- `force_download`/`local_files_only`: to reuse the cache or not
- `proxies`: configure HTTP session

When pushing models, similar parameters are supported:
- `commit_message`: custom commit message
- `private`: create a private repo if missing
- `create_pr`: create a PR instead of pushing to `main`
- `branch`: push to a branch instead of the `main` branch
- `allow_patterns`/`ignore_patterns`: filter which files to upload
- `token`
- ...

All of these parameters can be added to the implementations we saw above and passed to the `huggingface_hub` methods.
However, if a parameter changes or a new feature is added, you will need to update your package. Supporting those
parameters also means more documentation to maintain on your side. To see how to mitigate these limitations, let's jump
to our next section **class inheritance**.

## A more complex approach: class inheritance

As we saw above, there are two main methods to include in your library to integrate it with the Hub: upload files
(`push_to_hub`) and download files (`from_pretrained`). You can implement those methods by yourself but it comes with
caveats. To tackle this, `huggingface_hub` provides a tool that uses class inheritance. Let's see how it works!

In a lot of cases, a library already implements its model using a Python class. The class contains the properties of
the model and methods to load, run, train, and evaluate it. Our approach is to extend this class to include upload and
download features using mixins. A [Mixin](https://stackoverflow.com/a/547714) is a class that is meant to extend an
existing class with a set of specific features using multiple inheritance. `huggingface_hub` provides its own mixin,
the [`ModelHubMixin`]. The key here is to understand its behavior and how to customize it.

The [`ModelHubMixin`] class implements 3 *public* methods (`push_to_hub`, `save_pretrained` and `from_pretrained`). Those
are the methods that your users will call to load/save models with your library. [`ModelHubMixin`] also defines 2
*private* methods (`_save_pretrained` and `_from_pretrained`). Those are the ones you must implement. So to integrate
your library, you should:

1. Make your Model class inherit from [`ModelHubMixin`].
2. Implement the private methods:
    - [`~ModelHubMixin._save_pretrained`]: method taking as input a path to a directory and saving the model to it.
    You must write all the logic to dump your model in this method: model card, model weights, configuration files,
    training logs, and figures. Any relevant information for this model must be handled by this method.
    [Model Cards](https://huggingface.co/docs/hub/model-cards) are particularly important to describe your model. Check
    out [our implementation guide](./model-cards) for more details.
    - [`~ModelHubMixin._from_pretrained`]: **class method** taking as input a `model_id` and returning an instantiated
    model. The method must download the relevant files and load them.
3. You are done!

The advantage of using [`ModelHubMixin`] is that once you take care of the serialization/loading of the files, you are ready to go. You don't need to worry about stuff like repo creation, commits, PRs, or revisions. The [`ModelHubMixin`] also ensures public methods are documented and type annotated, and you'll be able to view your model's download count on the Hub. All of this is handled by the [`ModelHubMixin`] and available to your users.

### A concrete example: PyTorch

A good example of what we saw above is [`PyTorchModelHubMixin`], our integration for the PyTorch framework. This is a ready-to-use integration.

#### How to use it?

Here is how any user can load/save a PyTorch model from/to the Hub:

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin


# Define your Pytorch model exactly the same way you are used to
>>> class MyModel(
...         nn.Module,
...         PyTorchModelHubMixin, # multiple inheritance
...         library_name="keras-nlp",
...         tags=["keras"],
...         repo_url="https://github.com/keras-team/keras-nlp",
...         docs_url="https://keras.io/keras_nlp/",
...         # ^ optional metadata to generate model card
...     ):
...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
...         super().__init__()
...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
...         self.linear = nn.Linear(output_size, vocab_size)

...     def forward(self, x):
...         return self.linear(x + self.param)

# 1. Create model
>>> model = MyModel(hidden_size=128)

# Config is automatically created based on input + default values
>>> model.param.shape[0]
128

# 2. (optional) Save model to local directory
>>> model.save_pretrained("path/to/my-awesome-model")

# 3. Push model weights to the Hub
>>> model.push_to_hub("my-awesome-model")

# 4. Initialize model from the Hub => config has been preserved
>>> model = MyModel.from_pretrained("username/my-awesome-model")
>>> model.param.shape[0]
128

# Model card has been correctly populated
>>> from huggingface_hub import ModelCard
>>> card = ModelCard.load("username/my-awesome-model")
>>> card.data.tags
["keras", "pytorch_model_hub_mixin", "model_hub_mixin"]
>>> card.data.library_name
"keras-nlp"
```

#### Implementation

The implementation is actually very straightforward, and the full implementation can be found [here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py).

1. First, inherit your class from `ModelHubMixin`:

```python
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)
```

2. Implement the `_save_pretrained` method:

```py
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        save_model_as_safetensor(self.module, str(save_directory / SAFETENSORS_SINGLE_FILE))

```

3. Implement the `_from_pretrained` method:

```python
class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   @classmethod # Must be a classmethod!
   def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: str,
      cache_dir: str,
      force_download: bool,
      proxies: Optional[Dict],
      resume_download: bool,
      local_files_only: bool,
      token: Union[str, bool, None],
      map_location: str = "cpu", # additional argument
      strict: bool = False, # additional argument
      **model_kwargs,
   ):
      """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)

         model_file = hf_hub_download(
            repo_id=model_id,
            filename=SAFETENSORS_SINGLE_FILE,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
            )
         return cls._load_as_safetensor(model, model_file, map_location, strict)
```

And that's it! Your library now enables users to upload and download files to and from the Hub.

### Advanced usage

In the section above, we quickly discussed how the [`ModelHubMixin`] works. In this section, we will see some of its more advanced features to improve your library integration with the Hugging Face Hub.

#### Model card

[`ModelHubMixin`] generates the model card for you. Model cards are files that accompany the models and provide important information about them. Under the hood, model cards are simple Markdown files with additional metadata. Model cards are essential for discoverability, reproducibility, and sharing! Check out the [Model Cards guide](https://huggingface.co/docs/hub/model-cards) for more details.

Generating model cards semi-automatically is a good way to ensure that all models pushed with your library will share common metadata: `library_name`, `tags`, `license`, `pipeline_tag`, etc. This makes all models backed by your library easily searchable on the Hub and provides some resource links for users landing on the Hub. You can define the metadata directly when inheriting from [`ModelHubMixin`]:

```py
class UniDepthV1(
   nn.Module,
   PyTorchModelHubMixin,
   library_name="unidepth",
   repo_url="https://github.com/lpiccinelli-eth/UniDepth",
   docs_url=...,
   pipeline_tag="depth-estimation",
   license="cc-by-nc-4.0",
   tags=["monocular-metric-depth-estimation", "arxiv:1234.56789"]
):
   ...
```

By default, a generic model card will be generated with the info you've provided (example: [pyp1/VoiceCraft_giga830M](https://huggingface.co/pyp1/VoiceCraft_giga830M)). But you can define your own model card template as well!

In this example, all models pushed with the `VoiceCraft` class will automatically include a citation section and license details. For more details on how to define a model card template, please check the [Model Cards guide](./model-cards).

```py
MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This is a VoiceCraft model. For more details, please check out the official Github repo: https://github.com/jasonppy/VoiceCraft. This model is shared under a Attribution-NonCommercial-ShareAlike 4.0 International license.

## Citation

@article{peng2024voicecraft,
  author    = {Peng, Puyuan and Huang, Po-Yao and Li, Daniel and Mohamed, Abdelrahman and Harwath, David},
  title     = {VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild},
  journal   = {arXiv},
  year      = {2024},
}
"""

class VoiceCraft(
   nn.Module,
   PyTorchModelHubMixin,
   library_name="voicecraft",
   model_card_template=MODEL_CARD_TEMPLATE,
   ...
):
   ...
```


Finally, if you want to extend the model card generation process with dynamic values, you can override the [`~ModelHubMixin.generate_model_card`] method:

```py
from huggingface_hub import ModelCard, PyTorchModelHubMixin

class UniDepthV1(nn.Module, PyTorchModelHubMixin, ...):
   (...)

   def generate_model_card(self, *args, **kwargs) -> ModelCard:
      card = super().generate_model_card(*args, **kwargs)
      card.data.metrics = ...  # add metrics to the metadata
      card.text += ... # append section to the modelcard
      return card
```

#### Config

[`ModelHubMixin`] handles the model configuration for you. It automatically checks the input values when you instantiate the model and serializes them in a `config.json` file. This provides 2 benefits:
1. Users will be able to reload the model with the exact same parameters as you.
2. Having a `config.json` file automatically enables analytics on the Hub (i.e. the "downloads" count).

But how does it work in practice? Several rules make the process as smooth as possible from a user perspective:
- if your `__init__` method expects a `config` input, it will be automatically saved in the repo as `config.json`.
- if the `config` input parameter is annotated with a dataclass type (e.g. `config: Optional[MyConfigClass] = None`), then the `config` value will be correctly deserialized for you.
- all values passed at initialization will also be stored in the config file. This means you don't necessarily have to expect a `config` input to benefit from it.

Example:

```py
class MyModel(ModelHubMixin):
   def __init__(value: str, size: int = 3):
      self.value = value
      self.size = size

   (...) # implement _save_pretrained / _from_pretrained

model = MyModel(value="my_value")
model.save_pretrained(...)

# config.json contains passed and default values
{"value": "my_value", "size": 3}
```

But what if a value cannot be serialized as JSON? By default, the value will be ignored when saving the config file. However, in some cases your library already expects a custom object as input that cannot be serialized, and you don't want to update your internal logic to update its type. No worries! You can pass custom encoders/decoders for any type when inheriting from [`ModelHubMixin`]. This is a bit more work but ensures your internal logic is untouched when integrating your library with the Hub.

Here is a concrete example where a class expects a `argparse.Namespace` config as input:

```py
class VoiceCraft(nn.Module):
    def __init__(self, args):
      self.pattern = self.args.pattern
      self.hidden_size = self.args.hidden_size
      ...
```

One solution can be to update the `__init__` signature to `def __init__(self, pattern: str, hidden_size: int)` and update all snippets that instantiates your class. This is a perfectly valid way to fix it but it might break downstream applications using your library.

Another solution is to provide a simple encoder/decoder to convert `argparse.Namespace` to a dictionary.

```py
from argparse import Namespace

class VoiceCraft(
   nn.Module,
   PytorchModelHubMixin,  # inherit from mixin
   coders: {
      Namespace = (
         lambda x: vars(x),  # Encoder: how to convert a `Namespace` to a valid jsonable value?
         lambda data: Namespace(**data),  # Decoder: how to reconstruct a `Namespace` from a dictionary?
      )
   }
):
    def __init__(self, args: Namespace): # annotate `args`
      self.pattern = self.args.pattern
      self.hidden_size = self.args.hidden_size
      ...
```

In the snippet above, both the internal logic and the `__init__` signature of the class did not change. This means all existing code snippets for your library will continue to work. To achieve this, we had to:
1. Inherit from the mixin (`PytorchModelHubMixin` in this case).
2. Pass a `coders` parameter in the inheritance. This is a dictionary where keys are custom types you want to process. Values are a tuple `(encoder, decoder)`.
   - The encoder expects an object of the specified type as input and returns a jsonable value. This will be used when saving a model with `save_pretrained`.
   - The decoder expects raw data (typically a dictionary) as input and reconstructs the initial object. This will be used when loading the model with `from_pretrained`.
3. Add a type annotation to the `__init__` signature. This is important to let the mixin know which type is expected by the class and, therefore, which decoder to use.

For the sake of simplicity, the encoder/decoder functions in the example above are not robust. For a concrete implementation, you would most likely have to handle corner cases properly.

## Quick comparison

Let's quickly sum up the two approaches we saw with their advantages and drawbacks. The table below is only indicative.
Your framework might have some specificities that you need to address. This guide is only here to give guidelines and
ideas on how to handle integration. In any case, feel free to contact us if you have any questions!

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->
| Integration | Using helpers | Using [`ModelHubMixin`] |
|:---:|:---:|:---:|
| User experience | `model = load_from_hub(...)`<br>`push_to_hub(model, ...)` | `model = MyModel.from_pretrained(...)`<br>`model.push_to_hub(...)` |
| Flexibility | Very flexible.<br>You fully control the implementation. | Less flexible.<br>Your framework must have a model class. |
| Maintenance | More maintenance to add support for configuration, and new features. Might also require fixing issues reported by users. | Less maintenance as most of the interactions with the Hub are implemented in `huggingface_hub`. |
| Documentation / Type annotation | To be written manually. | Partially handled by `huggingface_hub`. |
| Download counter | To be handled manually. | Enabled by default if class has a `config` attribute. |
| Model card | To be handled manually | Generated by default with library_name, tags, etc. |
