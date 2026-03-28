<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Webhooks Server

Webhooks sind ein Grundpfeiler für MLOps-bezogene Funktionen. Sie ermöglichen es Ihnen, auf neue Änderungen in bestimmten Repos oder auf alle Repos, die bestimmten Benutzern/Organisationen gehören, die Sie interessieren, zu hören. Dieser Leitfaden erklärt, wie Sie den `huggingface_hub` nutzen können, um einen Server zu erstellen, der auf Webhooks hört und ihn in einen Space zu implementieren. Es wird davon ausgegangen, dass Sie mit dem Konzept der Webhooks auf dem Huggingface Hub vertraut sind. Um mehr über Webhooks selbst zu erfahren, können Sie zuerst diesen [Leitfaden](https://huggingface.co/docs/hub/webhooks) lesen.

Die Basis-Klasse, die wir in diesem Leitfaden verwenden werden, ist der [`WebhooksServer`]. Es handelt sich um eine Klasse, mit der sich ein Server leicht konfigurieren lässt, der Webhooks vom Huggingface Hub empfangen kann. Der Server basiert auf einer Gradio-App. Er verfügt über eine Benutzeroberfläche zur Anzeige von Anweisungen für Sie oder Ihre Benutzer und eine API zum Hören auf Webhooks.

> [!TIP]
> Um ein Beispiel eines laufenden Webhook-Servers zu sehen, werfen Sie einen Blick auf den [Spaces CI Bot](https://huggingface.co/spaces/spaces-ci-bot/webhook). Es handelt sich um einen Space, der kurzlebige Umgebungen startet, wenn ein PR in einem Space geöffnet wird.

> [!WARNING]
> Dies ist ein [experimentelles Feature](../package_reference/environment_variables#hfhubdisableexperimentalwarning). Das bedeutet, dass wir noch daran arbeiten, die API zu verbessern. Es könnten in der Zukunft ohne vorherige Ankündigung Änderungen vorgenommen werden. Stellen Sie sicher, dass Sie die Version des `huggingface_hub` in Ihren Anforderungen festlegen.


## Einen Endpunkt erstellen

Das Implementieren eines Webhook-Endpunkts ist so einfach wie das Dekorieren einer Funktion. Lassen Sie uns ein erstes Beispiel betrachten, um die Hauptkonzepte zu erklären:

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Einen Trainingsjob auslösen, wenn ein Datensatz aktualisiert wird
        ...
```

Speichern Sie diesen Ausschnitt in einer Datei namens `'app.py'` und führen Sie ihn mit `'python app.py'` aus. Sie sollten eine Nachricht wie diese sehen:

```text
Webhook secret is not defined. This means your webhook endpoints will be open to everyone.
To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization:
        `app = WebhooksServer(webhook_secret='my_secret', ...)`
For more details about webhook secrets, please refer to https://huggingface.co/docs/hub/webhooks#webhook-secret.
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://1fadb0f52d8bf825fc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces

Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
Go to https://huggingface.co/settings/webhooks to setup your webhooks.
```

Gute Arbeit! Sie haben gerade einen Webhook-Server gestartet! Lassen Sie uns genau aufschlüsseln, was passiert ist:

1. Durch das Dekorieren einer Funktion mit [`webhook_endpoint`] wurde im Hintergrund ein [`WebhooksServer`]-Objekt erstellt. Wie Sie sehen können, handelt es sich bei diesem Server um eine Gradio-App, die unter http://127.0.0.1:7860 läuft. Wenn Sie diese URL in Ihrem Browser öffnen, sehen Sie eine Landing Page mit Anweisungen zu den registrierten Webhooks.
2. Eine Gradio-App ist im Kern ein FastAPI-Server. Eine neue POST-Route `/webhooks/trigger_training` wurde hinzugefügt. Dies ist die Route, die auf Webhooks hört und die Funktion `trigger_training` ausführt, wenn sie ausgelöst wird. FastAPI wird das Payload automatisch parsen und es der Funktion als [`WebhookPayload`]-Objekt übergeben. Dies ist ein `pydantisches` Objekt, das alle Informationen über das Ereignis enthält, das den Webhook ausgelöst hat.
3. Die Gradio-App hat auch einen Tunnel geöffnet, um Anfragen aus dem Internet zu empfangen. Das Interessante daran ist: Sie können einen Webhook auf https://huggingface.co/settings/webhooks konfigurieren, der auf Ihren lokalen Rechner zeigt. Dies ist nützlich zum Debuggen Ihres Webhook-Servers und zum schnellen Iterieren, bevor Sie ihn in einem Space bereitstellen.
4. Schließlich teilen Ihnen die Logs auch mit, dass Ihr Server derzeit nicht durch ein Geheimnis gesichert ist. Dies ist für das lokale Debuggen nicht problematisch, sollte aber für später berücksichtigt werden.

> [!WARNING]
> Standardmäßig wird der Server am Ende Ihres Skripts gestartet. Wenn Sie es in einem Notizbuch ausführen, können Sie den Server manuell starten, indem Sie `decorated_function.run()` aufrufen. Da ein einzigartiger Server verwendet wird, müssen Sie den Server nur einmal starten, auch wenn Sie mehrere Endpunkte haben.


## Konfigurieren eines Webhook

Jetzt, da Sie einen Webhook-Server am Laufen haben, möchten Sie einen Webhook konfigurieren, um Nachrichten zu empfangen.
Gehen Sie zu https://huggingface.co/settings/webhooks, klicken Sie auf "Add a new webhook" und konfigurieren Sie Ihren Webhook. Legen Sie die Ziel-Repositories fest, die Sie beobachten möchten, und die Webhook-URL, hier `https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training`.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/configure_webhook.png"/>
</div>

Und das war's! Sie können den Webhook jetzt auslösen, indem Sie das Ziel-Repository aktualisieren (z.B. einen Commit pushen). Überprüfen Sie den Aktivitäts-Tab Ihres Webhooks, um die ausgelösten Ereignisse zu sehen. Jetzt, wo Sie eine funktionierende Einrichtung haben, können Sie sie testen und schnell iterieren. Wenn Sie Ihren Code ändern und den Server neu starten, könnte sich Ihre öffentliche URL ändern. Stellen Sie sicher, dass Sie die Webhook-Konfiguration im Hub bei Bedarf aktualisieren.

## Bereitstellung in einem Space

Jetzt, da Sie einen funktionierenden Webhook-Server haben, ist das Ziel, ihn in einem Space bereitzustellen. Gehen Sie zu https://huggingface.co/new-space, um einen Space zu erstellen. Geben Sie ihm einen Namen, wählen Sie das Gradio SDK und klicken Sie auf "Create Space". Laden Sie Ihren Code in den Space in einer Datei namens `app.py` hoch. Ihr Space wird automatisch gestartet! Für weitere Informationen zu Spaces lesen Sie bitte diesen [Leitfaden](https://huggingface.co/docs/hub/spaces-overview).

Ihr Webhook-Server läuft nun auf einem öffentlichen Space. In den meisten Fällen möchten Sie ihn mit einem Geheimnis absichern. Gehen Sie zu Ihren Space-Einstellungen > Abschnitt "Repository secrets" > "Add a secret". Setzen Sie die Umgebungsvariable `WEBHOOK_SECRET` auf den von Ihnen gewählten Wert. Gehen Sie zurück zu den [Webhook-Einstellungen](https://huggingface.co/settings/webhooks) und setzen Sie das Geheimnis in der Webhook-Konfiguration. Jetzt werden von Ihrem Server nur Anfragen mit dem korrekten Geheimnis akzeptiert.

Und das war's! Ihr Space ist nun bereit, Webhooks vom Hub zu empfangen. Bitte beachten Sie, dass wenn Sie den Space auf einer kostenlosen 'cpu-basic' Hardware ausführen, er nach 48 Stunden Inaktivität heruntergefahren wird. Wenn Sie einen permanenten Space benötigen, sollten Sie in Erwägung ziehen, auf eine [upgraded hardware](https://huggingface.co/docs/hub/spaces-gpus#hardware-specs) umzustellen.

## Erweiterte Nutzung

Der obenstehende Leitfaden erklärte den schnellsten Weg, einen [`WebhooksServer`] einzurichten. In diesem Abschnitt werden wir sehen, wie man ihn weiter anpassen kann.

### Mehrere Endpunkte

Sie können mehrere Endpunkte auf demselben Server registrieren. Beispielsweise möchten Sie vielleicht einen Endpunkt haben, um einen Trainingsjob auszulösen und einen anderen, um eine Modellevaluierung auszulösen. Dies können Sie tun, indem Sie mehrere `@webhook_endpoint`-Dekorateure hinzufügen:

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Einen Trainingsjob auslösen, wenn ein Datensatz aktualisiert wird
        ...

@webhook_endpoint
async def trigger_evaluation(payload: WebhookPayload) -> None:
    if payload.repo.type == "model" and payload.event.action == "update":
        # Einen Evaluierungsauftrag auslösen, wenn ein Modell aktualisiert wird
        ...
```

Dies wird zwei Endpunkte erstellen:

```text
(...)
Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_evaluation
```

### Benutzerdefinierter Server

Um mehr Flexibilität zu erhalten, können Sie auch direkt ein [`WebhooksServer`] Objekt erstellen. Dies ist nützlich, wenn Sie die Startseite Ihres Servers anpassen möchten. Sie können dies tun, indem Sie eine [Gradio UI](https://gradio.app/docs/#blocks) übergeben, die die Standard-UI überschreibt. Zum Beispiel können Sie Anweisungen für Ihre Benutzer hinzufügen oder ein Formular zur manuellen Auslösung der Webhooks hinzufügen. Bei der Erstellung eines [`WebhooksServer`] können Sie mit dem Dekorateur [`~WebhooksServer.add_webhook`] neue Webhooks registrieren.

Hier ist ein vollständiges Beispiel:

```python
import gradio as gr
from fastapi import Request
from huggingface_hub import WebhooksServer, WebhookPayload

# 1. Benutzerdefinierte UI definieren
with gr.Blocks() as ui:
    ...

# 2. Erstellen eines WebhooksServer mit benutzerdefinierter UI und Geheimnis
app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

# 3. Webhook mit explizitem Namen registrieren
@app.add_webhook("/say_hello")
async def hello(payload: WebhookPayload):
    return {"message": "hello"}

# 4. Webhook mit implizitem Namen registrierene
@app.add_webhook
async def goodbye(payload: WebhookPayload):
    return {"message": "goodbye"}

# 5. Server starten (optional)
app.run()
```

1. Wir definieren eine benutzerdefinierte UI mit Gradio-Blöcken. Diese UI wird auf der Startseite des Servers angezeigt.
2. Wir erstellen ein [`WebhooksServer`]-Objekt mit einer benutzerdefinierten UI und einem Geheimnis. Das Geheimnis ist optional und kann mit der `WEBHOOK_SECRET` Umgebungsvariable gesetzt werden.
3. Wir registrieren einen Webhook mit einem expliziten Namen. Dies wird einen Endpunkt unter `/webhooks/say_hello` erstellen.
4. Wir registrieren einen Webhook mit einem impliziten Namen. Dies wird einen Endpunkt unter `/webhooks/goodbye` erstellen.
5. Wir starten den Server. Dies ist optional, da Ihr Server automatisch am Ende des Skripts gestartet wird.
