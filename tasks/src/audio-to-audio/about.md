## Use cases

1. Speech Enhancement (Noise removal)

Speech Enhancement is a bit self explanatory. It improves (or enhances) the quality of an audio by removing noise. There are multiple libraries to solve this task, such as Speechbrain, Asteroid and ESPNet. Here is a simple example using Speechbrain 

```python
from speechbrain.pretrained import SpectralMaskEnhancement
model = SpectralMaskEnhancement.from_hparams(
  "speechbrain/mtl-mimic-voicebank"
)
model.enhance_file("file.wav")
```

Alternatively, you can use the [Inference API](https://huggingface.co/inference-api) to solve this task

```python
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/speechbrain/mtl-mimic-voicebank"

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = query("sample1.flac")
```

2. Audio Source Separation

Audio Source Separation allows you to isolate the different sounds from the individual sources. For example, if you have an audio file with multiple people speaking, you can get an audio file for each of them. You can then use an Automatic Speech Recognition system to extract the text from each of these sources as an initial step for your system!

Audio to audio can also be used to remove noise from audio files: you get one audio for the person that is speaking and another audio for the noise. This can also be helpful when you have an audio of multiple people with some noise: you can get an audio for each person and then an audio for the noise.

## Training a model for your own data

If you want to learn how to train models for audio to audio, we recommend the following tutorials:

- [https://speechbrain.github.io/tutorial_enhancement.html](https://speechbrain.github.io/tutorial_enhancement.html)
- [https://speechbrain.github.io/tutorial_separation.html](https://speechbrain.github.io/tutorial_separation.html)