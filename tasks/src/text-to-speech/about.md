The Hub contains over [100 TTS models](https://huggingface.co/models?pipeline_tag=text-to-speech&sort=downloads) that you can use right away by trying out the widgets directly in the browser or calling the models as a service using the Accelerated Inference API. Here is a simple code snippet to do exactly this

```python
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response

output = query({"inputs": "This is a test"})
```

You can also use libraries such as [espnet](https://huggingface.co/models?library=espnet&pipeline_tag=automatic-speech-recognition&sort=downloads) if you want to handle the Inference directly.

```python
from espnet2.bin.tts_inference import Text2Speech
    
model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

speech, *_ = model("text to generate speech from")
```