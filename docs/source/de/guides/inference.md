<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Inferenz auf Servern ausführen

Inferenz ist der Prozess, bei dem ein trainiertes Modell verwendet wird, um Vorhersagen für neue Daten zu treffen. Da dieser Prozess rechenintensiv sein kann, kann die Ausführung auf einem dedizierten Server eine interessante Option sein. Die `huggingface_hub` Bibliothek bietet eine einfache Möglichkeit, einen Dienst aufzurufen, der die Inferenz für gehostete Modelle durchführt. Es gibt mehrere Dienste, mit denen Sie sich verbinden können:
- [Inferenz API](https://huggingface.co/docs/api-inference/index): ein Service, der Ihnen ermöglicht, beschleunigte Inferenz auf der Infrastruktur von Hugging Face kostenlos auszuführen. Dieser Service ist eine schnelle Möglichkeit, um anzufangen, verschiedene Modelle zu testen und AI-Produkte zu prototypisieren.
- [Inferenz Endpunkte](https://huggingface.co/inference-endpoints/index): ein Produkt zur einfachen Bereitstellung von Modellen im Produktivbetrieb. Die Inferenz wird von Hugging Face in einer dedizierten, vollständig verwalteten Infrastruktur auf einem Cloud-Anbieter Ihrer Wahl durchgeführt.


> [!TIP]
> [`InferenceClient`] ist ein Python-Client, der HTTP-Anfragen an unsere APIs stellt. Wenn Sie die HTTP-Anfragen direkt mit Ihrem bevorzugten Tool (curl, postman,...) durchführen möchten, lesen Sie bitte die Dokumentationsseiten der [Inferenz API](https://huggingface.co/docs/api-inference/index) oder der [Inferenz Endpunkte](https://huggingface.co/docs/inference-endpoints/index).
>
> Für die Webentwicklung wurde ein [JS-Client](https://huggingface.co/docs/huggingface.js/inference/README) veröffentlicht. Wenn Sie sich für die Spieleentwicklung interessieren, sollten Sie einen Blick auf unser [C#-Projekt](https://github.com/huggingface/unity-api) werfen.

## Erste Schritte

Los geht's mit einer Text-zu-Bild-Aufgabe:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> image = client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")
```

Wir haben einen [`InferenceClient`] mit den Standardparametern initialisiert. Das Einzige, was Sie wissen müssen, ist die [Aufgabe](#unterstützte-aufgaben), die Sie ausführen möchten. Standardmäßig wird der Client sich mit der Inferenz API verbinden und ein Modell auswählen, um die Aufgabe abzuschließen. In unserem Beispiel haben wir ein Bild aus einem Textprompt generiert. Der zurückgegebene Wert ist ein `PIL.Image`-Objekt, das in eine Datei gespeichert werden kann.

> [!WARNING]
> Die API ist darauf ausgelegt, einfach zu sein. Nicht alle Parameter und Optionen sind für den Endbenutzer verfügbar oder beschrieben. Schauen Sie auf [dieser Seite](https://huggingface.co/docs/api-inference/detailed_parameters) nach, wenn Sie mehr über alle verfügbaren Parameter für jede Aufgabe erfahren möchten.

### Verwendung eines spezifischen Modells

Was ist, wenn Sie ein bestimmtes Modell verwenden möchten? Sie können es entweder als Parameter angeben oder direkt auf Instanzebene spezifizieren:

```python
>>> from huggingface_hub import InferenceClient
# Client für ein spezifisches Modell initialisieren
>>> client = InferenceClient(model="prompthero/openjourney-v4")
>>> client.text_to_image(...)
# Oder nutzen Sie einen generischen Client, geben aber Ihr Modell als Argument an
>>> client = InferenceClient()
>>> client.text_to_image(..., model="prompthero/openjourney-v4")
```

> [!TIP]
> Es gibt mehr als 200k Modelle im Hugging Face Hub! Jede Aufgabe im [`InferenceClient`] kommt mit einem empfohlenen Modell. Beachten Sie, dass die HF-Empfehlung sich im Laufe der Zeit ohne vorherige Ankündigung ändern kann. Daher ist es am besten, ein Modell explizit festzulegen, sobald Sie sich entschieden haben. In den meisten Fällen werden Sie daran interessiert sein, ein Modell zu finden, das speziell auf _Ihre_ Bedürfnisse zugeschnitten ist. Besuchen Sie die [Modelle](https://huggingface.co/models)-Seite im Hub, um Ihre Möglichkeiten zu erkunden.

### Verwendung einer spezifischen URL

Die oben gesehenen Beispiele nutzen die kostenfrei gehostete Inferenz API. Dies erweist sich als sehr nützlich für Prototyping und schnelles Testen. Wenn Sie bereit sind, Ihr Modell in die Produktion zu übernehmen, müssen Sie eine dedizierte Infrastruktur verwenden. Hier kommen [Inferenz Endpunkte](https://huggingface.co/docs/inference-endpoints/index) ins Spiel. Es ermöglicht Ihnen, jedes Modell zu implementieren und als private API freizugeben. Nach der Implementierung erhalten Sie eine URL, zu der Sie mit genau dem gleichen Code wie zuvor eine Verbindung herstellen können, wobei nur der `Modell`-Parameter geändert wird:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# oder
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

### Authentifizierung

Aufrufe, die mit dem [`InferenceClient`] gemacht werden, können mit einem [User Access Token](https://huggingface.co/docs/hub/security-tokens) authentifiziert werden. Standardmäßig wird das auf Ihrem Computer gespeicherte Token verwendet, wenn Sie angemeldet sind (sehen Sie hier, [wie Sie sich anmelden können](https://huggingface.co/docs/huggingface_hub/quick-start#login)). Wenn Sie nicht angemeldet sind, können Sie Ihr Token als Instanzparameter übergeben:

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(token="hf_***")
```

> [!TIP]
> Die Authentifizierung ist NICHT zwingend erforderlich, wenn Sie die Inferenz API verwenden. Authentifizierte Benutzer erhalten jedoch ein höheres kostenloses Kontingent, um mit dem Service zu arbeiten. Ein Token ist auch zwingend erforderlich, wenn Sie Inferenz auf Ihren privaten Modellen oder auf privaten Endpunkten ausführen möchten.

## Unterstützte Aufgaben

Das Ziel von [`InferenceClient`] ist es, die einfachste Schnittstelle zum Ausführen von Inferenzen auf Hugging Face-Modellen bereitzustellen. Es verfügt über eine einfache API, die die gebräuchlichsten Aufgaben unterstützt. Hier ist eine Liste der derzeit unterstützten Aufgaben:

| Domäne          | Aufgabe                                                                                       | Unterstützt | Dokumentation                                       |
| --------------- | --------------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------- |
| Audio           | [Audio Classification](https://huggingface.co/tasks/audio-classification)                     | ✅           | [`~InferenceClient.audio_classification`]           |
|                 | [Automatic Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition)     | ✅           | [`~InferenceClient.automatic_speech_recognition`]   |
|                 | [Text-to-Speech](https://huggingface.co/tasks/text-to-speech)                                 | ✅           | [`~InferenceClient.text_to_speech`]                 |
| Computer Vision | [Image Classification](https://huggingface.co/tasks/image-classification)                     | ✅           | [`~InferenceClient.image_classification`]           |
|                 | [Image Segmentation](https://huggingface.co/tasks/image-segmentation)                         | ✅           | [`~InferenceClient.image_segmentation`]             |
|                 | [Image-to-Image](https://huggingface.co/tasks/image-to-image)                                 | ✅           | [`~InferenceClient.image_to_image`]                 |
|                 | [Image-to-Text](https://huggingface.co/tasks/image-to-text)                                   | ✅           | [`~InferenceClient.image_to_text`]                  |
|                 | [Object Detection](https://huggingface.co/tasks/object-detection)                             | ✅           | [`~InferenceClient.object_detection`]               |
|                 | [Text-to-Image](https://huggingface.co/tasks/text-to-image)                                   | ✅           | [`~InferenceClient.text_to_image`]                  |
|                 | [Zero-Shot-Image-Classification](https://huggingface.co/tasks/zero-shot-image-classification) | ✅           | [`~InferenceClient.zero_shot_image_classification`] |
| Multimodal      | [Documentation Question Answering](https://huggingface.co/tasks/document-question-answering)  | ✅           | [`~InferenceClient.document_question_answering`]    |
|                 | [Visual Question Answering](https://huggingface.co/tasks/visual-question-answering)           | ✅           | [`~InferenceClient.visual_question_answering`]      |
| NLP             | [Conversational](https://huggingface.co/tasks/conversational)                                 | ✅           | [`~InferenceClient.conversational`]                 |
|                 | [Feature Extraction](https://huggingface.co/tasks/feature-extraction)                         | ✅           | [`~InferenceClient.feature_extraction`]             |
|                 | [Fill Mask](https://huggingface.co/tasks/fill-mask)                                           | ✅           | [`~InferenceClient.fill_mask`]                      |
|                 | [Question Answering](https://huggingface.co/tasks/question-answering)                         | ✅           | [`~InferenceClient.question_answering`]             |
|                 | [Sentence Similarity](https://huggingface.co/tasks/sentence-similarity)                       | ✅           | [`~InferenceClient.sentence_similarity`]            |
|                 | [Summarization](https://huggingface.co/tasks/summarization)                                   | ✅           | [`~InferenceClient.summarization`]                  |
|                 | [Table Question Answering](https://huggingface.co/tasks/table-question-answering)             | ✅           | [`~InferenceClient.table_question_answering`]       |
|                 | [Text Classification](https://huggingface.co/tasks/text-classification)                       | ✅           | [`~InferenceClient.text_classification`]            |
|                 | [Text Generation](https://huggingface.co/tasks/text-generation)                               | ✅           | [`~InferenceClient.text_generation`]                |
|                 | [Token Classification](https://huggingface.co/tasks/token-classification)                     | ✅           | [`~InferenceClient.token_classification`]           |
|                 | [Translation](https://huggingface.co/tasks/translation)                                       | ✅           | [`~InferenceClient.translation`]                    |
|                 | [Zero Shot Classification](https://huggingface.co/tasks/zero-shot-classification)             | ✅           | [`~InferenceClient.zero_shot_classification`]       |
| Tabular         | [Tabular Classification](https://huggingface.co/tasks/tabular-classification)                 | ✅           | [`~InferenceClient.tabular_classification`]         |
|                 | [Tabular Regression](https://huggingface.co/tasks/tabular-regression)                         | ✅           | [`~InferenceClient.tabular_regression`]             |


> [!TIP]
> Schauen Sie sich die [Aufgaben](https://huggingface.co/tasks)-Seite an, um mehr über jede Aufgabe zu erfahren, wie man sie verwendet und die beliebtesten Modelle für jede Aufgabe.

## Asynchroner Client

Eine asynchrone Version des Clients wird ebenfalls bereitgestellt, basierend auf `asyncio` und `aiohttp`. Sie können entweder `aiohttp` direkt installieren oder das `[inference]` Extra verwenden:

```sh
pip install aiohttp
# oder
pip install --upgrade huggingface_hub[inference]
```

Nach der Installation sind alle asynchronen API-Endpunkte über [`AsyncInferenceClient`] verfügbar. Seine Initialisierung und APIs sind genau gleich wie die synchronisierte Version.

```py
# Der Code muss in einem asyncio-konkurrenten Kontext ausgeführt werden.
# $ python -m asyncio
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> async for token in await client.text_generation("The Huggingface Hub is", stream=True):
...     print(token, end="")
 a platform for sharing and discussing ML-related content.
```

Für weitere Informationen zum `asyncio`-Modul konsultieren Sie bitte die [offizielle Dokumentation](https://docs.python.org/3/library/asyncio.html).


## Fortgeschrittene Tipps

Im obigen Abschnitt haben wir die Hauptaspekte von [`InferenceClient`] betrachtet. Lassen Sie uns in einige fortgeschrittene Tipps eintauchen.

### Zeitüberschreitung

Bei der Inferenz gibt es zwei Hauptursachen für eine Zeitüberschreitung:
- Der Inferenzprozess dauert lange, um abgeschlossen zu werden.
- Das Modell ist nicht verfügbar, beispielsweise wenn die Inferenz API es zum ersten Mal lädt.

Der [`InferenceClient`] verfügt über einen globalen Zeitüberschreitungsparameter (`timeout`), um diese beiden Aspekte zu behandeln. Standardmäßig ist er auf `None` gesetzt, was bedeutet, dass der Client unendlich lange auf den Abschluss der Inferenz warten wird. Wenn Sie mehr Kontrolle in Ihrem Arbeitsablauf wünschen, können Sie ihn auf einen bestimmten Wert in Sekunden setzen. Wenn die Zeitüberschreitungsverzögerung abläuft, wird ein [`InferenceTimeoutError`] ausgelöst. Sie können diesen Fehler abfangen und in Ihrem Code behandeln:

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### Binäre Eingaben

Einige Aufgaben erfordern binäre Eingaben, zum Beispiel bei der Arbeit mit Bildern oder Audiodateien. In diesem Fall versucht der [`InferenceClient] so permissiv wie möglich zu sein und akzeptiert verschiedene Typen:
- rohe `Bytes`
- ein Datei-ähnliches Objekt, geöffnet als Binär (`with open("audio.flac", "rb") as f: ...`)
- ein Pfad (`str` oder `Path`) zu einer lokalen Datei
- eine URL (`str`) zu einer entfernten Datei (z.B. `https://...`). In diesem Fall wird die Datei lokal heruntergeladen, bevor sie an die Inferenz API gesendet wird.

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```

