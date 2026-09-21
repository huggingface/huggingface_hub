<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# ಇನ್‌ಸ್ಟಾಲೇಶನ್

ಆರಂಭಿಸುವ ಮೊದಲು, ಸೂಕ್ತವಾದ ಪ್ಯಾಕೇಜ್‌ಗಳನ್ನು ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಿ ನಿಮ್ಮ ಪರಿಸರವನ್ನು ಸಿದ್ಧಪಡಿಸಿಕೊಳ್ಳಬೇಕು.

`huggingface_hub` ಅನ್ನು **Python 3.10+** ನಲ್ಲಿ ಪರೀಕ್ಷಿಸಲಾಗಿದೆ.

## pip ಮೂಲಕ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು

`huggingface_hub` ಅನ್ನು [ವರ್ಚುವಲ್ ಎನ್ವಿರಾನ್‌ಮೆಂಟ್](https://docs.python.org/3/library/venv.html) ನಲ್ಲಿ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು ಅತ್ಯಂತ ಸೂಕ್ತ.
Python ವರ್ಚುವಲ್ ಎನ್ವಿರಾನ್‌ಮೆಂಟ್‌ಗಳ ಪರಿಚಯ ಇಲ್ಲದಿದ್ದರೆ, ಈ [ಮಾರ್ಗದರ್ಶಿಯನ್ನು](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) ನೋಡಿ.
ವರ್ಚುವಲ್ ಎನ್ವಿರಾನ್‌ಮೆಂಟ್ ಬೇರೆ ಬೇರೆ ಪ್ರಾಜೆಕ್ಟ್‌ಗಳನ್ನು ನಿರ್ವಹಿಸುವುದನ್ನು ಸುಲಭಗೊಳಿಸುತ್ತದೆ ಮತ್ತು ಡಿಪೆಂಡೆನ್ಸಿಗಳ ನಡುವಿನ ಹೊಂದಾಣಿಕೆ ಸಮಸ್ಯೆಗಳನ್ನು ತಪ್ಪಿಸುತ್ತದೆ.

ಮೊದಲು ನಿಮ್ಮ ಪ್ರಾಜೆಕ್ಟ್ ಡೈರೆಕ್ಟರಿಯಲ್ಲಿ ಒಂದು ವರ್ಚುವಲ್ ಎನ್ವಿರಾನ್‌ಮೆಂಟ್ ರಚಿಸಿ:

```bash
python -m venv .venv
```

ವರ್ಚುವಲ್ ಎನ್ವಿರಾನ್‌ಮೆಂಟ್ ಅನ್ನು ಸಕ್ರಿಯಗೊಳಿಸಿ. Linux ಮತ್ತು macOS ನಲ್ಲಿ:

```bash
source .venv/bin/activate
```

Windows ನಲ್ಲಿ ವರ್ಚುವಲ್ ಎನ್ವಿರಾನ್‌ಮೆಂಟ್ ಸಕ್ರಿಯಗೊಳಿಸಲು:

```bash
.venv/Scripts/activate
```

ಈಗ [PyPi ರಿಜಿಸ್ಟ್ರಿಯಿಂದ](https://pypi.org/project/huggingface-hub/) `huggingface_hub` ಅನ್ನು ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಲು ಸಿದ್ಧರಾಗಿದ್ದೀರಿ:

```bash
pip install --upgrade huggingface_hub
```

ಮುಗಿದ ನಂತರ, [ಇನ್‌ಸ್ಟಾಲೇಶನ್ ಸರಿಯಾಗಿದೆಯೇ ಪರಿಶೀಲಿಸಿ](#check-installation).

### ಐಚ್ಛಿಕ ಡಿಪೆಂಡೆನ್ಸಿಗಳನ್ನು ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು

`huggingface_hub` ನ ಕೆಲವು ಡಿಪೆಂಡೆನ್ಸಿಗಳು [ಐಚ್ಛಿಕ](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies), ಏಕೆಂದರೆ `huggingface_hub` ನ ಮೂಲ ಫೀಚರ್‌ಗಳನ್ನು ಚಲಾಯಿಸಲು ಅವು ಬೇಕಾಗಿಲ್ಲ. ಆದರೆ ಈ ಐಚ್ಛಿಕ ಡಿಪೆಂಡೆನ್ಸಿಗಳನ್ನು ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡದಿದ್ದರೆ `huggingface_hub` ನ ಕೆಲವು ಫೀಚರ್‌ಗಳು ಲಭ್ಯವಿರುವುದಿಲ್ಲ.

ಐಚ್ಛಿಕ ಡಿಪೆಂಡೆನ್ಸಿಗಳನ್ನು `pip` ಮೂಲಕ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಬಹುದು:
```bash
# Install dependencies for both torch-specific and MCP-specific features.
pip install 'huggingface_hub[mcp,torch]'
```

`huggingface_hub` ನಲ್ಲಿರುವ ಐಚ್ಛಿಕ ಡಿಪೆಂಡೆನ್ಸಿಗಳ ಪಟ್ಟಿ ಇಲ್ಲಿದೆ:
- `fastai`, `torch`: ಫ್ರೇಮ್‌ವರ್ಕ್-ನಿರ್ದಿಷ್ಟ ಫೀಚರ್‌ಗಳನ್ನು ಚಲಾಯಿಸಲು ಬೇಕಾದ ಡಿಪೆಂಡೆನ್ಸಿಗಳು.
- `dev`: ಲೈಬ್ರರಿಗೆ ಕೊಡುಗೆ ನೀಡಲು ಬೇಕಾದ ಡಿಪೆಂಡೆನ್ಸಿಗಳು. ಇದರಲ್ಲಿ `testing` (ಟೆಸ್ಟ್‌ಗಳನ್ನು ಚಲಾಯಿಸಲು), `typing` (ಟೈಪ್ ಚೆಕರ್ ಚಲಾಯಿಸಲು) ಮತ್ತು `quality` (ಲಿಂಟರ್‌ಗಳನ್ನು ಚಲಾಯಿಸಲು) ಸೇರಿವೆ.



### ಸೋರ್ಸ್‌ನಿಂದ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು

ಕೆಲವು ಸಂದರ್ಭಗಳಲ್ಲಿ `huggingface_hub` ಅನ್ನು ನೇರವಾಗಿ ಸೋರ್ಸ್‌ನಿಂದ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು ಉಪಯುಕ್ತ.
ಇದರಿಂದ ಇತ್ತೀಚಿನ ಸ್ಥಿರ ಆವೃತ್ತಿಯ ಬದಲಿಗೆ ಅತ್ಯಂತ ಹೊಸದಾದ `main` ಆವೃತ್ತಿಯನ್ನು ಬಳಸಬಹುದು.
ಇತ್ತೀಚಿನ ಬೆಳವಣಿಗೆಗಳೊಂದಿಗೆ ನವೀಕೃತವಾಗಿರಲು `main` ಆವೃತ್ತಿ ಉಪಯುಕ್ತ — ಉದಾಹರಣೆಗೆ, ಕೊನೆಯ ಅಧಿಕೃತ ಬಿಡುಗಡೆಯ ನಂತರ
ಒಂದು ಬಗ್ ಸರಿಪಡಿಸಲಾಗಿದ್ದರೂ ಹೊಸ ಬಿಡುಗಡೆ ಇನ್ನೂ ಹೊರಬಂದಿಲ್ಲದ ಸಂದರ್ಭದಲ್ಲಿ.

ಆದರೆ ಇದರರ್ಥ `main` ಆವೃತ್ತಿ ಯಾವಾಗಲೂ ಸ್ಥಿರವಾಗಿರದಿರಬಹುದು. `main` ಆವೃತ್ತಿಯನ್ನು ಕೆಲಸ ಮಾಡುವ ಸ್ಥಿತಿಯಲ್ಲಿ
ಇಡಲು ನಾವು ಪ್ರಯತ್ನಿಸುತ್ತೇವೆ, ಮತ್ತು ಹೆಚ್ಚಿನ ಸಮಸ್ಯೆಗಳು ಸಾಮಾನ್ಯವಾಗಿ ಕೆಲವು ಗಂಟೆಗಳಲ್ಲಿ ಅಥವಾ ಒಂದು ದಿನದಲ್ಲಿ
ಪರಿಹಾರವಾಗುತ್ತವೆ. ನಿಮಗೆ ಏನಾದರೂ ಸಮಸ್ಯೆ ಎದುರಾದರೆ ದಯವಿಟ್ಟು ಒಂದು ಇಶ್ಯೂ ತೆರೆಯಿರಿ — ಇನ್ನೂ ಬೇಗ ಸರಿಪಡಿಸಲು
ನಮಗೆ ಸಾಧ್ಯವಾಗುತ್ತದೆ!

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

ಸೋರ್ಸ್‌ನಿಂದ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವಾಗ ನಿರ್ದಿಷ್ಟ ಬ್ರಾಂಚ್ ಅನ್ನೂ ಸೂಚಿಸಬಹುದು. ಇನ್ನೂ ಮರ್ಜ್ ಆಗದ ಹೊಸ ಫೀಚರ್ ಅಥವಾ
ಹೊಸ ಬಗ್-ಫಿಕ್ಸ್ ಅನ್ನು ಪರೀಕ್ಷಿಸಲು ಇದು ಉಪಯುಕ್ತ:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

ಮುಗಿದ ನಂತರ, [ಇನ್‌ಸ್ಟಾಲೇಶನ್ ಸರಿಯಾಗಿದೆಯೇ ಪರಿಶೀಲಿಸಿ](#check-installation).

### ಎಡಿಟಬಲ್ ಇನ್‌ಸ್ಟಾಲ್

ಸೋರ್ಸ್‌ನಿಂದ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದರಿಂದ [ಎಡಿಟಬಲ್ ಇನ್‌ಸ್ಟಾಲ್](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) ಅನ್ನು ಸಿದ್ಧಪಡಿಸಿಕೊಳ್ಳಬಹುದು.
`huggingface_hub` ಗೆ ಕೊಡುಗೆ ನೀಡುವ ಯೋಜನೆ ಇದ್ದು ಕೋಡ್‌ನಲ್ಲಿ ಮಾಡಿದ ಬದಲಾವಣೆಗಳನ್ನು ಪರೀಕ್ಷಿಸಬೇಕಾದರೆ ಬಳಸುವ
ಸ್ವಲ್ಪ ಮುಂದುವರಿದ ಇನ್‌ಸ್ಟಾಲೇಶನ್ ವಿಧಾನ ಇದು. ಇದಕ್ಕಾಗಿ `huggingface_hub` ನ ಒಂದು ಸ್ಥಳೀಯ ಪ್ರತಿಯನ್ನು ನಿಮ್ಮ
ಮೆಷಿನ್‌ಗೆ ಕ್ಲೋನ್ ಮಾಡಿಕೊಳ್ಳಬೇಕು.

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

ಈ ಕಮಾಂಡ್‌ಗಳು ನೀವು ರಿಪೊಸಿಟರಿಯನ್ನು ಕ್ಲೋನ್ ಮಾಡಿದ ಫೋಲ್ಡರ್ ಅನ್ನು ನಿಮ್ಮ Python ಲೈಬ್ರರಿ ಪಾತ್‌ಗಳಿಗೆ ಜೋಡಿಸುತ್ತವೆ.
ಇನ್ನು ಮುಂದೆ Python ಸಾಮಾನ್ಯ ಲೈಬ್ರರಿ ಪಾತ್‌ಗಳ ಜೊತೆಗೆ ನೀವು ಕ್ಲೋನ್ ಮಾಡಿದ ಫೋಲ್ಡರ್‌ನಲ್ಲೂ ಹುಡುಕುತ್ತದೆ.
ಉದಾಹರಣೆಗೆ, ನಿಮ್ಮ Python ಪ್ಯಾಕೇಜ್‌ಗಳು ಸಾಮಾನ್ಯವಾಗಿ `./.venv/lib/python3.13/site-packages/` ನಲ್ಲಿ
ಇನ್‌ಸ್ಟಾಲ್ ಆಗುತ್ತಿದ್ದರೆ, Python `./huggingface_hub/` ಫೋಲ್ಡರ್‌ನಲ್ಲೂ ಹುಡುಕುತ್ತದೆ.

## Hugging Face CLI ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು

ನಿಮ್ಮ Python ಪರಿಸರವನ್ನು ಮುಟ್ಟದೆಯೇ `hf` CLI ಅನ್ನು ಸಿದ್ಧಪಡಿಸಲು ನಮ್ಮ ಒಂದೇ-ಸಾಲಿನ ಇನ್‌ಸ್ಟಾಲರ್‌ಗಳನ್ನು ಬಳಸಿ:

macOS ಮತ್ತು Linux ನಲ್ಲಿ:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Windows ನಲ್ಲಿ:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

ಈಗಾಗಲೇ ಇರುವ ಇನ್‌ಸ್ಟಾಲ್ ಅನ್ನು ಅಪ್‌ಗ್ರೇಡ್ ಮಾಡಲು `hf update` ಚಲಾಯಿಸಿ — `hf` ಅನ್ನು ಹೇಗೆ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಲಾಗಿತ್ತು (ಸ್ವತಂತ್ರ ಇನ್‌ಸ್ಟಾಲರ್, Homebrew, ಅಥವಾ pip) ಎಂಬುದನ್ನು ಅದು ಪತ್ತೆಹಚ್ಚಿ ಸೂಕ್ತ ಕಮಾಂಡ್ ಚಲಾಯಿಸುತ್ತದೆ.

## conda ಮೂಲಕ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡುವುದು

conda ನಿಮಗೆ ಹೆಚ್ಚು ಪರಿಚಿತವಾಗಿದ್ದರೆ, [conda-forge ಚಾನೆಲ್](https://anaconda.org/conda-forge/huggingface_hub) ಬಳಸಿ `huggingface_hub` ಅನ್ನು ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಬಹುದು:


```bash
conda install -c conda-forge huggingface_hub
```

ಮುಗಿದ ನಂತರ, [ಇನ್‌ಸ್ಟಾಲೇಶನ್ ಸರಿಯಾಗಿದೆಯೇ ಪರಿಶೀಲಿಸಿ](#check-installation).

<a id="check-installation"></a> <!-- keep the English anchor working for #check-installation links -->

## ಇನ್‌ಸ್ಟಾಲೇಶನ್ ಪರಿಶೀಲನೆ

ಇನ್‌ಸ್ಟಾಲ್ ಆದ ಮೇಲೆ, `huggingface_hub` ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದೆಯೇ ಎಂದು ಈ ಕಮಾಂಡ್ ಚಲಾಯಿಸಿ ಪರಿಶೀಲಿಸಿ:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

ಈ ಕಮಾಂಡ್ [gpt2](https://huggingface.co/gpt2) ಮಾಡೆಲ್ ಬಗ್ಗೆ ಹಬ್‌ನಿಂದ ಮಾಹಿತಿಯನ್ನು ತರುತ್ತದೆ.
ಔಟ್‌ಪುಟ್ ಹೀಗಿರಬೇಕು:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows ಮಿತಿಗಳು

ಎಲ್ಲೆಡೆ ಉತ್ತಮ ML ಅನ್ನು ಎಲ್ಲರಿಗೂ ತಲುಪಿಸುವ ಗುರಿಯೊಂದಿಗೆ, `huggingface_hub` ಅನ್ನು ನಾವು ಕ್ರಾಸ್-ಪ್ಲಾಟ್‌ಫಾರ್ಮ್
ಲೈಬ್ರರಿಯಾಗಿ — ವಿಶೇಷವಾಗಿ Unix-ಆಧಾರಿತ ಮತ್ತು Windows ಎರಡೂ ವ್ಯವಸ್ಥೆಗಳಲ್ಲಿ ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡುವಂತೆ —
ನಿರ್ಮಿಸಿದ್ದೇವೆ. ಆದರೂ Windows ನಲ್ಲಿ ಚಲಾಯಿಸಿದಾಗ `huggingface_hub` ಗೆ ಕೆಲವು ಮಿತಿಗಳಿವೆ. ತಿಳಿದಿರುವ
ಸಮಸ್ಯೆಗಳ ಸಂಪೂರ್ಣ ಪಟ್ಟಿ ಇಲ್ಲಿದೆ. ದಾಖಲಾಗಿರದ ಯಾವುದೇ ಸಮಸ್ಯೆ ಎದುರಾದರೆ
[GitHub ನಲ್ಲಿ ಒಂದು ಇಶ್ಯೂ ತೆರೆದು](https://github.com/huggingface/huggingface_hub/issues/new/choose) ನಮಗೆ ತಿಳಿಸಿ.

- ಹಬ್‌ನಿಂದ ಡೌನ್‌ಲೋಡ್ ಆದ ಫೈಲ್‌ಗಳನ್ನು ಸಮರ್ಥವಾಗಿ ಕ್ಯಾಶ್ ಮಾಡಲು `huggingface_hub` ನ ಕ್ಯಾಶ್ ವ್ಯವಸ್ಥೆ
ಸಿಮ್‌ಲಿಂಕ್‌ಗಳನ್ನು ಅವಲಂಬಿಸಿದೆ. Windows ನಲ್ಲಿ ಸಿಮ್‌ಲಿಂಕ್‌ಗಳನ್ನು ಬಳಸಲು ನೀವು ಡೆವಲಪರ್ ಮೋಡ್ ಅನ್ನು
ಸಕ್ರಿಯಗೊಳಿಸಬೇಕು ಅಥವಾ ನಿಮ್ಮ ಸ್ಕ್ರಿಪ್ಟ್ ಅನ್ನು ಅಡ್ಮಿನ್ ಆಗಿ ಚಲಾಯಿಸಬೇಕು. ಅವು ಸಕ್ರಿಯವಾಗಿಲ್ಲದಿದ್ದರೂ
ಕ್ಯಾಶ್ ವ್ಯವಸ್ಥೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ, ಆದರೆ ಅಷ್ಟು ಸಮರ್ಥವಾಗಿ ಅಲ್ಲ. ಹೆಚ್ಚಿನ ವಿವರಗಳಿಗಾಗಿ
[ಕ್ಯಾಶ್ ಮಿತಿಗಳು](./guides/manage-cache#limitations) ವಿಭಾಗವನ್ನು ಓದಿ.
- ಹಬ್‌ನಲ್ಲಿರುವ ಫೈಲ್‌ಪಾತ್‌ಗಳಲ್ಲಿ ವಿಶೇಷ ಅಕ್ಷರಗಳಿರಬಹುದು (ಉದಾ. `"path/to?/my/file"`). Windows
[ವಿಶೇಷ ಅಕ್ಷರಗಳ](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)
ವಿಷಯದಲ್ಲಿ ಹೆಚ್ಚು ಕಟ್ಟುನಿಟ್ಟಾಗಿರುವುದರಿಂದ ಅಂತಹ ಫೈಲ್‌ಗಳನ್ನು Windows ನಲ್ಲಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಲು ಸಾಧ್ಯವಿಲ್ಲ.
ಅದೃಷ್ಟವಶಾತ್ ಇದು ಅಪರೂಪದ ಸಂದರ್ಭ. ಇದು ತಪ್ಪು ಎಂದು ನಿಮಗೆ ಅನಿಸಿದರೆ ರಿಪೊ ಮಾಲೀಕರನ್ನು ಸಂಪರ್ಕಿಸಿ, ಅಥವಾ
ಪರಿಹಾರ ಕಂಡುಕೊಳ್ಳಲು ನಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಿ.


## ಮುಂದಿನ ಹಂತಗಳು

`huggingface_hub` ನಿಮ್ಮ ಮೆಷಿನ್‌ನಲ್ಲಿ ಸರಿಯಾಗಿ ಇನ್‌ಸ್ಟಾಲ್ ಆದ ಮೇಲೆ, ನೀವು
[ಎನ್ವಿರಾನ್‌ಮೆಂಟ್ ವೇರಿಯಬಲ್‌ಗಳನ್ನು ಕಾನ್ಫಿಗರ್ ಮಾಡಬಹುದು](package_reference/environment_variables) ಅಥವಾ
ಆರಂಭಿಸಲು [ನಮ್ಮ ಮಾರ್ಗದರ್ಶಿಗಳಲ್ಲಿ ಒಂದನ್ನು ನೋಡಬಹುದು](guides/overview).
