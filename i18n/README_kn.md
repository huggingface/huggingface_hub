<p align="center">

&#x20; <picture>

&#x20;   <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface\_hub-dark.svg">

&#x20;   <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface\_hub.svg">

&#x20;   <img alt="huggingface\_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface\_hub.svg" width="352" height="59" style="max-width: 100%">

&#x20; </picture>

&#x20; <br/>

&#x20; <br/>

</p>



<p align="center">

&#x20;   <i>Huggingface Hub ನ ಅಧಿಕೃತ Python ಕ್ಲೈಂಟ್.</i>

</p>



<p align="center">

&#x20;   <a href="https://huggingface.co/docs/huggingface\_hub/en/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/huggingface\_hub/index.svg?down\_color=red\&down\_message=offline\&up\_message=online\&label=doc"></a>

&#x20;   <a href="https://github.com/huggingface/huggingface\_hub/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/huggingface\_hub.svg"></a>

&#x20;   <a href="https://github.com/huggingface/huggingface\_hub"><img alt="PyPi version" src="https://img.shields.io/pypi/pyversions/huggingface\_hub.svg"></a>

&#x20;   <a href="https://pypi.org/project/huggingface-hub"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/huggingface\_hub"></a>

&#x20;   <a href="https://codecov.io/gh/huggingface/huggingface\_hub"><img alt="Code coverage" src="https://codecov.io/gh/huggingface/huggingface\_hub/branch/main/graph/badge.svg?token=RXP95LE2XL"></a>

</p>



<h4 align="center">

&#x20;   <p>

&#x20;       <a href="https://github.com/huggingface/huggingface\_hub/blob/main/README.md">English</a> |

&#x20;       <a href="https://github.com/huggingface/huggingface\_hub/blob/main/i18n/README\_de.md">Deutsch</a> |

&#x20;       <a href="https://github.com/huggingface/huggingface\_hub/blob/main/i18n/README\_fr.md">Français</a> |

&#x20;       <a href="https://github.com/huggingface/huggingface\_hub/blob/main/i18n/README\_hi.md">हिंदी</a> |

&#x20;       <a href="https://github.com/huggingface/huggingface\_hub/blob/main/i18n/README\_ko.md">한국어</a> |

&#x20;       <a href="https://github.com/huggingface/huggingface\_hub/blob/main/i18n/README\_cn.md">中文（简体）</a> |

&#x20;       <b>ಕನ್ನಡ</b>

&#x20;   <p>

</h4>



\---



\*\*ದಸ್ತಾವೇಜು\*\*: <a href="https://hf.co/docs/huggingface\_hub" target="\_blank">https://hf.co/docs/huggingface\_hub</a>



\*\*ಮೂಲ ಕೋಡ್\*\*: <a href="https://github.com/huggingface/huggingface\_hub" target="\_blank">https://github.com/huggingface/huggingface\_hub</a>



\---



\## huggingface\_hub ಗ್ರಂಥಾಲಯಕ್ಕೆ ಸ್ವಾಗತ



`huggingface\_hub` ಗ್ರಂಥಾಲಯವು \[Hugging Face Hub](https://huggingface.co/) ನೊಂದಿಗೆ ಸಂವಹನ ನಡೆಸಲು ಅನುವು ಮಾಡಿಕೊಡುತ್ತದೆ — ಇದು ತೆರೆದ-ಮೂಲ ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಅನ್ನು ಸೃಷ್ಟಿಕರ್ತರು ಮತ್ತು ಸಹಯೋಗಿಗಳಿಗಾಗಿ ಪ್ರಜಾಪ್ರಭುತ್ವೀಕರಣಗೊಳಿಸುವ ವೇದಿಕೆ. ನಿಮ್ಮ ಯೋಜನೆಗಳಿಗಾಗಿ ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಮಾದರಿಗಳು ಮತ್ತು ಡೇಟಾಸೆಟ್‌ಗಳನ್ನು ಅನ್ವೇಷಿಸಿ. `huggingface\_hub` ಗ್ರಂಥಾಲಯವು Python ನಲ್ಲಿ ಇವೆಲ್ಲವನ್ನೂ ಮಾಡಲು ಸರಳ ಮಾರ್ಗವನ್ನು ಒದಗಿಸುತ್ತದೆ.



\## ಪ್ರಮುಖ ವೈಶಿಷ್ಟ್ಯಗಳು



\- Hub ನಿಂದ \[ಫೈಲ್‌ಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/download).

\- Hub ಗೆ \[ಫೈಲ್‌ಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/upload).

\- \[ನಿಮ್ಮ ರೆಪೊಸಿಟರಿಗಳನ್ನು ನಿರ್ವಹಿಸಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/repository).

\- ನಿಯೋಜಿತ ಮಾದರಿಗಳ ಮೇಲೆ \[ಅನುಮಾನ ಚಲಾಯಿಸಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/inference).

\- ಮಾದರಿಗಳು, ಡೇಟಾಸೆಟ್‌ಗಳು ಮತ್ತು Spaces ಗಾಗಿ \[ಹುಡುಕಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/search).

\- ನಿಮ್ಮ ಮಾದರಿಗಳನ್ನು ದಾಖಲಿಸಲು \[ಮಾದರಿ ಕಾರ್ಡ್‌ಗಳನ್ನು ಹಂಚಿಕೊಳ್ಳಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/model-cards).

\- PR ಗಳು ಮತ್ತು ಕಾಮೆಂಟ್‌ಗಳ ಮೂಲಕ \[ಸಮುದಾಯದೊಂದಿಗೆ ತೊಡಗಿಸಿಕೊಳ್ಳಿ](https://huggingface.co/docs/huggingface\_hub/en/guides/community).



\## ಸ್ಥಾಪನೆ



`pip` ನೊಂದಿಗೆ `huggingface\_hub` ಪ್ಯಾಕೇಜ್ ಅನ್ನು ಸ್ಥಾಪಿಸಿ:



```bash

pip install huggingface\_hub

```



\## ತ್ವರಿತ ಪ್ರಾರಂಭ



\### ಫೈಲ್‌ಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ



ಒಂದೇ ಫೈಲ್ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ:



```py

from huggingface\_hub import hf\_hub\_download



hf\_hub\_download(repo\_id="tiiuae/falcon-7b-instruct", filename="config.json")

```



ಅಥವಾ ಸಂಪೂರ್ಣ ರೆಪೊಸಿಟರಿ:



```py

from huggingface\_hub import snapshot\_download



snapshot\_download("stabilityai/stable-diffusion-2-1")

```



\### ಲಾಗಿನ್



```bash

hf auth login

```



\### ರೆಪೊಸಿಟರಿ ರಚಿಸಿ



```py

from huggingface\_hub import create\_repo



create\_repo(repo\_id="super-cool-model")

```



\### ಫೈಲ್‌ಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ



```py

from huggingface\_hub import upload\_file



upload\_file(

&#x20;   path\_or\_fileobj="/home/lysandre/dummy-test/README.md",

&#x20;   path\_in\_repo="README.md",

&#x20;   repo\_id="lysandre/test-model",

)

```



\## ಕೊಡುಗೆಗಳು ಸ್ವಾಗತಾರ್ಹ 🤗



ಎಲ್ಲರೂ ಕೊಡುಗೆ ನೀಡಲು ಸ್ವಾಗತ. ಕೋಡ್ ಮಾತ್ರವಲ್ಲ — ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸುವುದು, ದಾಖಲೆಗಳನ್ನು ಸುಧಾರಿಸುವುದು ಸಹ ಮೌಲ್ಯಯುತ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ \[ಕೊಡುಗೆ ಮಾರ್ಗದರ್ಶಿ](https://github.com/huggingface/huggingface\_hub/blob/main/CONTRIBUTING.md) ನೋಡಿ.

