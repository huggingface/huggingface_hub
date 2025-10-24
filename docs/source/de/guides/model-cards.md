<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Erstellen und Teilen von Model Cards

Die `huggingface_hub`-Bibliothek bietet eine Python-Schnittstelle zum Erstellen, Teilen und Aktualisieren von Model Cards. Besuchen Sie [die spezielle Dokumentationsseite](https://huggingface.co/docs/hub/models-cards) für einen tieferen Einblick in das, was Model Cards im Hub sind und wie sie unter der Haube funktionieren.

> [!TIP]
> [Neu (Beta)! Probieren Sie unsere experimentelle Model Card Creator App aus](https://huggingface.co/spaces/huggingface/Model_Cards_Writing_Tool)

## Eine Model Card vom Hub laden

Um eine bestehende Karte vom Hub zu laden, können Sie die Funktion [`ModelCard.load`] verwenden. Hier laden wir die Karte von [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans).

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

Diese Karte hat einige nützliche Attribute, auf die Sie zugreifen oder die Sie nutzen möchten:

- `card.data`: Gibt eine [`ModelCardData`]-Instanz mit den Metadaten der Model Card zurück. Rufen Sie `.to_dict()` auf diese Instanz auf, um die Darstellung als Wörterbuch zu erhalten.
- `card.text`: Gibt den Textinhalt der Karte *ohne den Metadatenkopf* zurück.
- `card.content`: Gibt den Textinhalt der Karte, *einschließlich des Metadatenkopfes*, zurück.

## Model Cards erstellen

### Aus Text

Um eine Model Card aus Text zu initialisieren, übergeben Sie einfach den Textinhalt der Karte an `ModelCard` beim Initialisieren.

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

Eine andere Möglichkeit besteht darin, dies mit f-Strings zu tun. Im folgenden Beispiel:

- Verwenden wir [`ModelCardData.to_yaml`], um die von uns definierten Metadaten in YAML umzuwandeln, damit wir sie in die Model Card einfügen können.
- Zeigen wir, wie Sie eine Vorlagenvariable über Python f-Strings verwenden könnten.

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

Das obige Beispiel würde uns eine Karte hinterlassen, die so aussieht:

```
---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)
```

### Aus einem Jinja-Template

Wenn Sie `Jinja2` installiert haben, können Sie Model Cards aus einer Jinja-Vorlagendatei erstellen. Schauen wir uns ein einfaches Beispiel an:

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Definieren Sie Ihre Jinja-Vorlage
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Schreiben Sie die Vorlage in eine Datei
Path('custom_template.md').write_text(template_text)

# Definieren Sie die Metadaten der Karte
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Erstellen Sie eine Karte aus der Vorlage und übergeben Sie dabei alle gewünschten Jinja-Vorlagenvariablen.
# In unserem Fall übergeben wir "author"
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

Das resultierende Karten-Markdown sieht so aus:

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

Wenn Sie Daten in card.data aktualisieren, wird dies in der Karte selbst widergespiegelt.

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

Jetzt, wie Sie sehen können, wurde der Metadatenkopf aktualisiert:

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

Wenn Sie die Karteninformationen aktualisieren, können Sie durch Aufrufen von [`ModelCard.validate`] überprüfen, ob die Karte immer noch gültig für den Hub ist. Dies stellt sicher, dass die Karte alle Validierungsregeln erfüllt, die im Hugging Face Hub eingerichtet wurden.

### Aus dem Standard-Template

Anstatt Ihr eigenes Template zu verwenden, können Sie auch das [Standard-Template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md) verwenden, welches eine vollständig ausgestattete Model Card mit vielen Abschnitten ist, die Sie vielleicht ausfüllen möchten. Unter der Haube verwendet es [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/), um eine Vorlagendatei auszufüllen.

> [!TIP]
> Beachten Sie, dass Sie Jinja2 installiert haben müssen, um `from_template` zu verwenden. Sie können dies mit `pip install Jinja2` tun.

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

## Model Cards teilen

Wenn Sie mit dem Hugging Face Hub authentifiziert sind (entweder durch Verwendung von `hf auth login` oder [`login`]), können Sie Karten zum Hub hinzufügen, indem Sie einfach [`ModelCard.push_to_hub`] aufrufen. Schauen wir uns an, wie das funktioniert...

Zuerst erstellen wir ein neues Repo namens 'hf-hub-modelcards-pr-test' im Namensraum des authentifizierten Benutzers:

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

Dann erstellen wir eine Karte aus der Standardvorlage (genau wie die oben definierte):

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

Schließlich laden wir das zum Hub hoch:

```python
card.push_to_hub(repo_id)
```

Sie können die resultierende Karte [hier](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md) überprüfen.

Wenn Sie eine Karte als Pull-Request hinzufügen möchten, können Sie beim Aufruf von `push_to_hub` einfach `create_pr=True` angeben:

```python
card.push_to_hub(repo_id, create_pr=True)
```

Ein PR, der mit diesem Befehl erstellt wurde, kann [hier](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3) aufgerufen werden.

### Evaluierungsergebnisse einbeziehen

Um Evaluierungsergebnisse in den Metadaten `model-index` einzufügen, können Sie ein [`EvalResult`] oder eine Liste von `EvalResult` mit Ihren zugehörigen Evaluierungsergebnissen übergeben. Im Hintergrund wird der `model-index` erstellt, wenn Sie `card.data.to_dict()` aufrufen. Weitere Informationen darüber, wie dies funktioniert, finden Sie in [diesem Abschnitt der Hub-Dokumentation](https://huggingface.co/docs/hub/models-cards#evaluation-results).

> [!TIP]
> Beachten Sie, dass die Verwendung dieser Funktion erfordert, dass Sie das Attribut `model_name` in [`ModelCardData`] einbeziehen.

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

Die resultierende `card.data` sollte so aussehen:

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
Wenn Sie mehr als ein Evaluierungsergebnis teilen möchten, übergeben Sie einfach eine Liste von `EvalResult`:

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

Dies sollte Ihnen die folgenden `card.data` hinterlassen:

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
