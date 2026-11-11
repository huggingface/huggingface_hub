# Streamlit-Chat-Beispiel

Dieses eigenständige Beispiel zeigt einen einfachen Chat mit Sitzungsverlauf und einer Platzhalter-Antwort. Es verändert die Abhängigkeiten der Hugging Face Hub-Bibliothek nicht.

## Streamlit-Oberfläche

```bash
cd examples/streamlit_chat
python -m pip install -r requirements.txt
streamlit run app.py
```

## API für Postman

Starte in einem zweiten Terminal aus diesem Verzeichnis:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Erstelle in Postman einen Request mit diesen Einstellungen:

- Methode: `POST`
- URL: `http://127.0.0.1:8000/chat`
- Header: `Content-Type: application/json`
- Body (raw, JSON):

```json
{
	"message": "Hallo!"
}
```

Alternativ kannst du die Datei `postman_collection.json` direkt in Postman importieren.

Die Antwort sieht so aus:

```json
{
	"answer": "Das ist eine Test-Antwort auf: Hallo!"
}
```

Der Status-Endpunkt ist unter `http://127.0.0.1:8000/health` erreichbar. Die Chat-Antwort ist noch ein Platzhalter und kann später durch einen Modellaufruf ersetzt werden.