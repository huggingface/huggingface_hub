<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <i>Huggingface Hub ನ ಅಧಿಕೃತ Python ಕ್ಲೈಂಟ್.</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/en/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface_hub/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>
    <a href="https://github.com/huggingface/huggingface_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface_hub.svg"></a>
    <a href="https://github.com/huggingface/huggingface_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface_hub.svg"></a>
    <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface_hub"></a>
    <a href="https://codecov.io/gh/huggingface/huggingface_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文（简体）</a> |
        <b>ಕನ್ನಡ</b>
    <p>
</h4>

---

**ದಸ್ತಾವೇಜು**: <a href="https://hf.co/docs/huggingface_hub" target="_blank">https://hf.co/docs/huggingface_hub</a>

**ಮೂಲ ಕೋಡ್**: <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub</a>

---

## huggingface_hub ಗ್ರಂಥಾಲಯಕ್ಕೆ ಸ್ವಾಗತ

`huggingface_hub` ಗ್ರಂಥಾಲಯವು [Hugging Face Hub](https://huggingface.co/) ನೊಂದಿಗೆ ಸಂವಹನ ನಡೆಸಲು ಅನುವು ಮಾಡಿಕೊಡುತ್ತದೆ — ಇದು ತೆರೆದ-ಮೂಲ ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಅನ್ನು ಸೃಷ್ಟಿಕರ್ತರು ಮತ್ತು ಸಹಯೋಗಿಗಳಿಗಾಗಿ ಪ್ರಜಾಪ್ರಭುತ್ವೀಕರಣಗೊಳಿಸುವ ವೇದಿಕೆ. ನಿಮ್ಮ ಯೋಜನೆಗಳಿಗಾಗಿ ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಮಾದರಿಗಳು ಮತ್ತು ಡೇಟಾಸೆಟ್‌ಗಳನ್ನು ಅನ್ವೇಷಿಸಿ ಅಥವಾ Hub ನಲ್ಲಿ ಹೋಸ್ಟ್ ಮಾಡಲಾದ ಸಾವಿರಾರು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಅಪ್ಲಿಕೇಶನ್‌ಗಳೊಂದಿಗೆ ಆಡಿ. ನೀವು ನಿಮ್ಮ ಸ್ವಂತ ಮಾದರಿಗಳು, ಡೇಟಾಸೆಟ್‌ಗಳು ಮತ್ತು ಡೆಮೊಗಳನ್ನು ಸಮುದಾಯದೊಂದಿಗೆ ರಚಿಸಿ ಹಂಚಿಕೊಳ್ಳಬಹುದು. `huggingface_hub` ಗ್ರಂಥಾಲಯವು Python ನಲ್ಲಿ ಇವೆಲ್ಲವನ್ನೂ ಮಾಡಲು ಸರಳ ಮಾರ್ಗವನ್ನು ಒದಗಿಸುತ್ತದೆ.

## ಪ್ರಮುಖ ವೈಶಿಷ್ಟ್ಯಗಳು

- Hub ನಿಂದ [ಫೈಲ್‌ಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ](https://huggingface.co/docs/huggingface_hub/en/guides/download).
- Hub ಗೆ [ಫೈಲ್‌ಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
- [ನಿಮ್ಮ ರೆಪೊಸಿಟರಿಗಳನ್ನು ನಿರ್ವಹಿಸಿ](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
- ನಿಯೋಜಿತ ಮಾದರಿಗಳ ಮೇಲೆ [ಅನುಮಾನ ಚಲಾಯಿಸಿ](https://huggingface.co/docs/huggingface_hub/en/guides/inference).
- ಮಾದರಿಗಳು, ಡೇಟಾಸೆಟ್‌ಗಳು ಮತ್ತು Spaces ಗಾಗಿ [ಹುಡುಕಿ](https://huggingface.co/docs/huggingface_hub/en/guides/search).
- ನಿಮ್ಮ ಮಾದರಿಗಳನ್ನು ದಾಖಲಿಸಲು [ಮಾದರಿ ಕಾರ್ಡ್‌ಗಳನ್ನು ಹಂಚಿಕೊಳ್ಳಿ](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards).
- PR ಗಳು ಮತ್ತು ಕಾಮೆಂಟ್‌ಗಳ ಮೂಲಕ [ಸಮುದಾಯದೊಂದಿಗೆ ತೊಡಗಿಸಿಕೊಳ್ಳಿ](https://huggingface.co/docs/huggingface_hub/en/guides/community).

## ಸ್ಥಾಪನೆ

[pip](https://pypi.org/project/huggingface-hub/) ನೊಂದಿಗೆ `huggingface_hub` ಪ್ಯಾಕೇಜ್ ಅನ್ನು ಸ್ಥಾಪಿಸಿ:

```bash
pip install huggingface_hub
```

ವೇಗವಾದ ಮತ್ತು ವಿಶ್ವಾಸಾರ್ಹ ಸ್ಥಾಪನೆಗಾಗಿ [`uv`](https://docs.astral.sh/uv/) ಬಳಸಲು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ:

```bash
uv pip install huggingface_hub
```

ಐಚ್ಛಿಕ ಅವಲಂಬನೆಗಳೊಂದಿಗೆ ಸ್ಥಾಪಿಸಲು, ಉದಾಹರಣೆಗೆ MCP ಮಾಡ್ಯೂಲ್ ಬಳಸಲು:

```bash
pip install "huggingface_hub[mcp]"
```

ಸ್ಥಾಪನೆ ಮತ್ತು ಐಚ್ಛಿಕ ಅವಲಂಬನೆಗಳ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ [ಸ್ಥಾಪನೆ ಮಾರ್ಗದರ್ಶಿ](https://huggingface.co/docs/huggingface_hub/en/installation) ನೋಡಿ.

## ತ್ವರಿತ ಪ್ರಾರಂಭ

### ಫೈಲ್‌ಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ

ಒಂದೇ ಫೈಲ್ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ:

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

ಅಥವಾ ಸಂಪೂರ್ಣ ರೆಪೊಸಿಟರಿ:

```py
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

ಫೈಲ್‌ಗಳು ಸ್ಥಳೀಯ ಕ್ಯಾಶ್ ಫೋಲ್ಡರ್‌ನಲ್ಲಿ ಡೌನ್‌ಲೋಡ್ ಆಗುತ್ತವೆ. ಹೆಚ್ಚಿನ ವಿವರಗಳಿಗಾಗಿ [ಈ ಮಾರ್ಗದರ್ಶಿ](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) ನೋಡಿ.

### ಲಾಗಿನ್

Hugging Face Hub ಅಪ್ಲಿಕೇಶನ್‌ಗಳನ್ನು ದೃಢೀಕರಿಸಲು ಟೋಕನ್‌ಗಳನ್ನು ಬಳಸುತ್ತದೆ (ನೋಡಿ [ದಸ್ತಾವೇಜು](https://huggingface.co/docs/hub/security-tokens)). ನಿಮ್ಮ ಯಂತ್ರದಲ್ಲಿ ಲಾಗಿನ್ ಮಾಡಲು:

```bash
hf auth login
# ಅಥವಾ ಪರಿಸರ ವೇರಿಯಬಲ್ ಬಳಸಿ
hf auth login --token $HUGGINGFACE_TOKEN
```

### ರೆಪೊಸಿಟರಿ ರಚಿಸಿ

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### ಫೈಲ್‌ಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ

ಒಂದೇ ಫೈಲ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ:

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

ಅಥವಾ ಸಂಪೂರ್ಣ ಫೋಲ್ಡರ್:

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

ವಿವರಗಳಿಗಾಗಿ [ಅಪ್‌ಲೋಡ್ ಮಾರ್ಗದರ್ಶಿ](https://huggingface.co/docs/huggingface_hub/en/guides/upload) ನೋಡಿ.

## Hub ನೊಂದಿಗೆ ಸಂಯೋಜನೆ

ನಾವು ತೆರೆದ-ಮೂಲ ML ಗ್ರಂಥಾಲಯಗಳೊಂದಿಗೆ ಪಾಲುದಾರಿಕೆ ಮಾಡಿ ಉಚಿತ ಮಾದರಿ ಹೋಸ್ಟಿಂಗ್ ಮತ್ತು ಆವೃತ್ತಿ ನಿರ್ವಹಣೆ ಒದಗಿಸುತ್ತೇವೆ. ಅಸ್ತಿತ್ವದಲ್ಲಿರುವ ಸಂಯೋಜನೆಗಳನ್ನು [ಇಲ್ಲಿ](https://huggingface.co/docs/hub/libraries) ಕಾಣಬಹುದು.

ಪ್ರಯೋಜನಗಳು:

- ಗ್ರಂಥಾಲಯಗಳು ಮತ್ತು ಅವರ ಬಳಕೆದಾರರಿಗೆ ಉಚಿತ ಮಾದರಿ ಅಥವಾ ಡೇಟಾಸೆಟ್ ಹೋಸ್ಟಿಂಗ್.
- git ಆಧಾರಿತ ವಿಧಾನದ ಮೂಲಕ ಅಂತರ್ನಿರ್ಮಿತ ಫೈಲ್ ಆವೃತ್ತಿ ನಿರ್ವಹಣೆ, ದೊಡ್ಡ ಫೈಲ್‌ಗಳಿಗೂ ಸಹ.
- ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಮಾದರಿಗಳೊಂದಿಗೆ ಆಡಲು ಬ್ರೌಸರ್ ವಿಜೆಟ್‌ಗಳು.
- ಯಾರಾದರೂ ನಿಮ್ಮ ಗ್ರಂಥಾಲಯಕ್ಕಾಗಿ ಹೊಸ ಮಾದರಿ ಅಪ್‌ಲೋಡ್ ಮಾಡಬಹುದು.
- ವೇಗದ ಡೌನ್‌ಲೋಡ್‌ಗಳು! ನಾವು Cloudfront (CDN) ಬಳಸಿ ಡೌನ್‌ಲೋಡ್‌ಗಳನ್ನು ಭೌಗೋಳಿಕವಾಗಿ ಪ್ರತಿಕೃತಿ ಮಾಡುತ್ತೇವೆ.
- ಬಳಕೆ ಅಂಕಿಅಂಶಗಳು ಮತ್ತು ಇನ್ನಷ್ಟು ವೈಶಿಷ್ಟ್ಯಗಳು ಬರಲಿವೆ.

ನಿಮ್ಮ ಗ್ರಂಥಾಲಯವನ್ನು ಸಂಯೋಜಿಸಲು ಬಯಸಿದರೆ, ಚರ್ಚೆ ಪ್ರಾರಂಭಿಸಲು ಒಂದು issue ತೆರೆಯಿರಿ. ನಾವು ❤️ ನೊಂದಿಗೆ [ಹಂತ-ಹಂತದ ಮಾರ್ಗದರ್ಶಿ](https://huggingface.co/docs/hub/adding-a-library) ಬರೆದಿದ್ದೇವೆ.

## ಕೊಡುಗೆಗಳು (ವೈಶಿಷ್ಟ್ಯ ವಿನಂತಿಗಳು, ದೋಷಗಳು, ಇತ್ಯಾದಿ) ತುಂಬಾ ಸ್ವಾಗತಾರ್ಹ 💙💚💛💜🧡❤️

ಎಲ್ಲರೂ ಕೊಡುಗೆ ನೀಡಲು ಸ್ವಾಗತ, ಮತ್ತು ನಾವು ಪ್ರತಿಯೊಬ್ಬರ ಕೊಡುಗೆಯನ್ನು ಗೌರವಿಸುತ್ತೇವೆ. ಕೋಡ್ ಮಾತ್ರ ಸಮುದಾಯಕ್ಕೆ ಸಹಾಯ ಮಾಡುವ ಏಕೈಕ ಮಾರ್ಗವಲ್ಲ. ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸುವುದು, ಇತರರಿಗೆ ಸಹಾಯ ಮಾಡುವುದು, ಸಂಪರ್ಕ ಸಾಧಿಸುವುದು ಮತ್ತು ದಾಖಲೆಗಳನ್ನು ಸುಧಾರಿಸುವುದು ಸಮುದಾಯಕ್ಕೆ ಅಮೂಲ್ಯವಾಗಿದೆ. ಈ ರೆಪೊಸಿಟರಿಗೆ ಕೊಡುಗೆ ನೀಡಲು ಹೇಗೆ ಪ್ರಾರಂಭಿಸಬೇಕು ಎಂಬುದನ್ನು ಸಂಕ್ಷೇಪಿಸಲು ನಾವು [ಕೊಡುಗೆ ಮಾರ್ಗದರ್ಶಿ](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) ಬರೆದಿದ್ದೇವೆ.