<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Den Hub durchsuchen

In diesem Tutorial lernen Sie, wie Sie Modelle, Datensätze und Spaces auf dem Hub mit `huggingface_hub` durchsuchen können.

## Wie listet man Repositories auf?

Die `huggingface_hub`-Bibliothek enthält einen HTTP-Client [`HfApi`], um mit dem Hub zu interagieren.
Unter anderem kann er Modelle, Datensätze und Spaces auflisten, die auf dem Hub gespeichert sind:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

Die Ausgabe von [`list_models`] ist ein Iterator über die auf dem Hub gespeicherten Modelle.

Ähnlich können Sie [`list_datasets`] verwenden, um Datensätze aufzulisten und [`list_spaces`], um Spaces aufzulisten.

## Wie filtert man Repositories?

Das Auflisten von Repositories ist großartig, aber jetzt möchten Sie vielleicht Ihre Suche filtern.
Die List-Helfer haben mehrere Attribute wie:
- `filter`
- `author`
- `search`
- ...

Zwei dieser Parameter sind intuitiv (`author` und `search`), aber was ist mit diesem `filter`?
`filter` nimmt als Eingabe ein [`ModelFilter`]-Objekt (oder [`DatasetFilter`]) entgegen.
Sie können es instanziieren, indem Sie angeben, welche Modelle Sie filtern möchten.

Hier ist ein Beispiel, um alle Modelle auf dem Hub zu erhalten, die Bildklassifizierung durchführen,
auf dem Imagenet-Datensatz trainiert wurden und mit PyTorch laufen.
Das kann mit einem einzigen [`ModelFilter`] erreicht werden. Attribute werden als "logisches UND" kombiniert.

```py
models = hf_api.list_models(
    filter=ModelFilter(
		task="image-classification",
		library="pytorch",
		trained_dataset="imagenet"
	)
)
```

Während des Filterns können Sie auch die Modelle sortieren und nur die Top-Ergebnisse abrufen.
Zum Beispiel holt das folgende Beispiel die 5 am häufigsten heruntergeladenen Datensätze auf dem Hub:

```py
>>> list(list_datasets(sort="downloads", direction=-1, limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```



Eine andere Möglichkeit, dies zu tun,
besteht darin, die [Modelle](https://huggingface.co/models) und [Datensätze](https://huggingface.co/datasets) Seiten
in Ihrem Browser zu besuchen, nach einigen Parametern zu suchen und die Werte in der URL anzusehen.
