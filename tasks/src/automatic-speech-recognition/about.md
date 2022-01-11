The Hub contains over [500 ASR models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads) that you can use right away by trying out the widgets directly in the browser or calling the models as a service using the Accelerated Inference API. Here is a simple code snippet to do exactly this:

```python
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = query("sample1.flac")
```

You can also use libraries such as [transformers](https://huggingface.co/models?library=transformers&pipeline_tag=automatic-speech-recognition&sort=downloads), [speechbrain](https://huggingface.co/models?library=speechbrain&pipeline_tag=automatic-speech-recognition&sort=downloads) and [espnet](https://huggingface.co/models?library=espnet&pipeline_tag=automatic-speech-recognition&sort=downloads) if you want to handle the Inference directly.

```python
from transformers import pipeline

with open("sample.flac", "rb") as f:
  data = f.read()

pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
pipe("sample.flac")
# {'text': "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOL ROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"}
```

## Solving ASR for your own data

We have some great news! You can do fine-tuning (transfer learning) to train a well-performing model without requiring as much data. Pretrained models such as Wav2Vec2 and HuBERT exist. [Facebook's Wav2Vec2 XLS-R model](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) is a large multilingual model trained on 128 languages and with 436K hours of speech.

The following detailed [blog post](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) shows how to fine-tune a pre-trained network on labeled data for ASR. This is easily done by adding a single layer on top of the pretrained network. We suggest to read the article for more info!

## Hugging Face XLSR-Wav2Vec2 Sprint

On March 2020, over 300 participants collaborated, trained and shared 236 ASR models in dozens of different languages. You can compare these models thanks to the [PapersWithCode](https://paperswithcode.com/dataset/common-voice) integration (see [Portuguese models](https://paperswithcode.com/sota/speech-recognition-on-common-voice-portuguese) for example).

![Leaderboard of ASR Models](/tasks/assets/automatic-speech-recognition/wav2vec2.png)

These events help democratize ASR for all languages, including low-resource languages. In addition to the trained models, the event helps to build practical collaborative knowledge.
