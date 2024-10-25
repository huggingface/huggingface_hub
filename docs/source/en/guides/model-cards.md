<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Create and share Model Cards

The `huggingface_hub` library provides a Python interface to create, share, and update Model Cards.
Visit [the dedicated documentation page](https://huggingface.co/docs/hub/models-cards)
for a deeper view of what Model Cards on the Hub are, and how they work under the hood.

<Tip>

[New (beta)! Try our experimental Model Card Creator App](https://huggingface.co/spaces/huggingface/Model_Cards_Writing_Tool)

</Tip>

## Load a Model Card from the Hub

To load an existing card from the Hub, you can use the [`ModelCard.load`] function. Here, we'll load the card from [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans).

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

This card has some helpful attributes that you may want to access/leverage:
  - `card.data`: Returns a [`ModelCardData`] instance with the model card's metadata. Call `.to_dict()` on this instance to get the representation as a dictionary.
  - `card.text`: Returns the text of the card, *excluding the metadata header*.
  - `card.content`: Returns the text content of the card, *including the metadata header*.

## Create Model Cards

### From Text

To initialize a Model Card from text, just pass the text content of the card to the `ModelCard` on init.

```python
content = """
---
language: en
license: mit
---

# My Model Card
"""

card = ModelCard(content)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True
```

Another way you might want to do this is with f-strings. In the following example, we:

- Use [`ModelCardData.to_yaml`] to convert metadata we defined to YAML so we can use it to insert the YAML block in the model card.
- Show how you might use a template variable via Python f-strings.

```python
card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
content = f"""
---
{ card_data.to_yaml() }
---

# My Model Card

This model created by [@{example_template_var}](https://github.com/{example_template_var})
"""

card = ModelCard(content)
print(card)
```

The above example would leave us with a card that looks like this:

```
---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)
```

### From a Jinja Template

If you have `Jinja2` installed, you can create Model Cards from a jinja template file. Let's see a basic example:

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Define your jinja template
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Write the template to a file
Path('custom_template.md').write_text(template_text)

# Define card metadata
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Create card from template, passing it any jinja template variables you want.
# In our case, we'll pass author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

The resulting card's markdown looks like this:

```
---
language: en
license: mit
library_name: keras
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

If you update any card.data, it'll reflect in the card itself.

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

Now, as you can see, the metadata header has been updated:

```
---
language: fr
license: apache-2.0
library_name: timm
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

As you update the card data, you can validate the card is still valid against the Hub by calling [`ModelCard.validate`]. This ensures that the card passes any validation rules set up on the Hugging Face Hub.

### From the Default Template

Instead of using your own template, you can also use the [default template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md), which is a fully featured model card with tons of sections you may want to fill out. Under the hood, it uses [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) to fill out a template file.

<Tip>

Note that you will have to have Jinja2 installed to use `from_template`. You can do so with `pip install Jinja2`.

</Tip>

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
card.save('my_model_card_2.md')
print(card)
```

## Share Model Cards

If you're authenticated with the Hugging Face Hub (either by using `huggingface-cli login` or [`login`]), you can push cards to the Hub by simply calling [`ModelCard.push_to_hub`]. Let's take a look at how to do that...

First, we'll create a new repo called 'hf-hub-modelcards-pr-test' under the authenticated user's namespace:

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

Then, we'll create a card from the default template (same as the one defined in the section above):

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
```

Finally, we'll push that up to the hub

```python
card.push_to_hub(repo_id)
```

You can check out the resulting card [here](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md).

If you instead wanted to push a card as a pull request, you can just say `create_pr=True` when calling `push_to_hub`:

```python
card.push_to_hub(repo_id, create_pr=True)
```

A resulting PR created from this command can be seen [here](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3).

## Update metadata

In this section we will see what metadata are in repo cards and how to update them.

`metadata` refers to a hash map (or key value) context that provides some high-level information about a model, dataset or Space. That information can include details such as the model's `pipeline type`, `model_id` or `model_description`. For more detail you can take a look to these guides: [Model Card](https://huggingface.co/docs/hub/model-cards#model-card-metadata), [Dataset Card](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) and [Spaces Settings](https://huggingface.co/docs/hub/spaces-settings#spaces-settings).
Now lets see some examples on how to update those metadata.


Let's start with a first example:

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

With these two lines of code you will update the metadata to set a new `pipeline_tag`.

By default, you cannot update a key that is already existing on the card. If you want to do so, you must pass
`overwrite=True` explicitly:


```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```

It often happen that you want to suggest some changes to a repository
on which you don't have write permission. You can do that by creating a PR on that repo which will allow the owners to
review and merge your suggestions.

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## Include Evaluation Results

To include evaluation results in the metadata `model-index`, you can pass an [`EvalResult`] or a list of `EvalResult` with your associated evaluation results. Under the hood it'll create the `model-index` when you call `card.data.to_dict()`. For more information on how this works, you can check out [this section of the Hub docs](https://huggingface.co/docs/hub/models-cards#evaluation-results).

<Tip>

Note that using this function requires you to include the `model_name` attribute in [`ModelCardData`].

</Tip>

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = EvalResult(
        task_type='image-classification',
        dataset_type='beans',
        dataset_name='Beans',
        metric_type='accuracy',
        metric_value=0.7
    )
)

card = ModelCard.from_template(card_data)
print(card.data)
```

The resulting `card.data` should look like this:

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
```

If you have more than one evaluation result you'd like to share, just pass a list of `EvalResult`:

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = [
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.7
        ),
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='f1',
            metric_value=0.65
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data
```

Which should leave you with the following `card.data`:

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65
```
