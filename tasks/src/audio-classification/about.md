## Use Cases

### Command Recognition

Command recognition or keyword spotting classifies utterances into a predefined set of commands. This is often done on-device for fast response time.

As an example, using the Google Speech Commands dataset, given an input, a model can classify which of the following commands the user is typing:

```
'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence'
```

Speechbrain models can easily perform this task with just a couple of lines of code!

```python
from speechbrain.pretrained import EncoderClassifier
model = EncoderClassifier.from_hparams(
  "speechbrain/google_speech_command_xvector"
)
model.classify_file("file.wav")
```

### Language Identification

Datasets such as VoxLingua107 allow anyone to train language identification models for up to 107 languages! This can be extremely useful as a preprocessing step for other systems. Here's an example [model](https://huggingface.co/TalTechNLP/voxlingua107-epaca-tdnn)trained on VoxLingua107.

### Emotion recognition

Emotion recognition is self explanatory. In addition to trying the widgets, you can use the Inference API to perform audio classification. Here is a simple example that uses a [HuBERT](https://huggingface.co/superb/hubert-large-superb-er) model fine-tuned for this task.

```python
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/superb/hubert-large-superb-er"

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = query("sample1.flac")
# [{'label': 'neu', 'score': 0.60},
# {'label': 'hap', 'score': 0.20},
# {'label': 'ang', 'score': 0.13},
# {'label': 'sad', 'score': 0.07}]
```

### Speaker Identification

Speaker Identification is classifying the audio of the person speaking. Speakers are usually predefined. You can try out this task with [this model](https://huggingface.co/superb/wav2vec2-base-superb-sid). A useful dataset for this task is VoxCeleb1.

## Solving audio classification for your own data

We have some great news! You can do fine-tuning (transfer learning) to train a well-performing model without requiring as much data. Pretrained models such as Wav2Vec2 and HuBERT exist. [Facebook's Wav2Vec2 XLS-R model](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) is a large multilingual model trained on 128 languages and with 436K hours of speech.

We suggest checking out the following [example](https://github.com/huggingface/transformers/tree/master/examples/pytorch/audio-classification) ([Colab Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/audio_classification.ipynb)) to learn how to fine-tune a model for audio classification with a single or multiple GPUs and share it on the Hub.
