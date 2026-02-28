<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg">
    <img alt="huggingface_hub library logo" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/huggingface_hub.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p> 

<p align="center">
    <i>The official Python client for the Huggingface Hub.</i>
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
        <b>English</b> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文（简体）</a>
    <p>
</h4>

---

**Документация:**: <a href="https://hf.co/docs/huggingface_hub" target="_blank">https://hf.co/docs/huggingface_hub</a>

**Исходный код**: <a href="https://github.com/huggingface/huggingface_hub" target="_blank">https://github.com/huggingface/huggingface_hub</a>

---

## Добро пожаловать в библиотеку huggingface_hub

Библиотека `huggingface_hub` — это способ удобно взаимодействовать с [Hugging Face Hub](https://huggingface.co/)платформой, которая делает машинное обучение доступным каждому. Здесь можно находить предобученные модели и датасеты, запускать тысячи ML-приложений прямо из браузера или делиться своими наработками с сообществом. Всё это — прямо из Python!

## Основные возможности

- [Скачивание файлов](https://huggingface.co/docs/huggingface_hub/en/guides/download) из хаба.
- [Загрузка своих файлов](https://huggingface.co/docs/huggingface_hub/en/guides/upload) из хаба.
- [Управление своими репозиториями](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
- [Запуск инференса](https://huggingface.co/docs/huggingface_hub/en/guides/inference) на размещённых моделях.
- [Поиск](https://huggingface.co/docs/huggingface_hub/en/guides/search) моделей, датасетов и Spaces
- [Публикация Model Cards](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) — описаний своих моделей.
- [Участие в жизни сообщества:](https://huggingface.co/docs/huggingface_hub/en/guides/community) пулл-реквесты, комментарии и т.д.

## Установка

Установить `huggingface_hub` можно через [pip](https://pypi.org/project/huggingface-hub/):

```bash
pip install huggingface_hub
```

Если удобнее — можно поставить через [conda](https://huggingface.co/docs/huggingface_hub/en/installation#install-with-conda).

По умолчанию пакет минимален, но для определённых задач можно поставить дополнительные зависимости. Например, чтобы полностью использовать возможности инференса:

```bash
pip install huggingface_hub[inference]
```

Подробнее об установке и зависимостях — в [installation guide](https://huggingface.co/docs/huggingface_hub/en/installation).

## Быстрый старт

### Скачиваем файлы

Скачать один файл:

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

Или сразу весь репозиторий:

```py
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1")
```

Файлы будут кэшироваться локально. Подробнее об этом — [this guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache).

### Вход в аккаунт

Для работы с хабом используется токен (см. [docs](https://huggingface.co/docs/hub/security-tokens)). Чтобы залогиниться в терминале:

```bash
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Создаём репозиторий

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### Загружаем файлы

Загружаем один файл:

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

Или целую папку:

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

Больше про это — [ в руководстве по загрузке](https://huggingface.co/docs/huggingface_hub/en/guides/upload).

## Интеграция с Hugging Face Hub

Hugging Face сотрудничает с разными библиотеками с открытым исходным кодом, чтобы обеспечить бесплатный хостинг и версионирование моделей. Список уже поддерживаемых интеграций —  [здесь](https://huggingface.co/docs/hub/libraries).

Что дают такие интеграции:

- Бесплатный хостинг моделей и датасетов — для библиотек и их пользователей.
- Версионирование файлов (включая большие) с помощью Git.
- Серверлесс-инференс для любых публичных моделей.
- Веб-виджеты для запуска моделей прямо в браузере.
- Кто угодно может загрузить новую модель для твоей библиотеки — достаточно просто добавить нужный тег, чтобы модель была видна и находилась в поиске.
- Скачивание происходит молниеносно! Мы используем Cloudfront (это CDN), который распределяет файлы по разным регионам, так что загрузка быстрая из любой точки мира.
- Статистика использования и другие фишки.

Хочешь интегрировать свою библиотеку? Просто создай issue — и обсудим! Есть даже [гайд по интеграции](https://huggingface.co/docs/hub/adding-a-library) — написан с любовью ❤️

## Мы рады любым контрибуциям 💙💚💛💜🧡❤️

Любой человек может внести вклад — и это очень ценно. Помощь — это не только код.
Можно отвечать на вопросы, помогать другим, улучшать документацию — всё это важно для комьюнити.
Вот [гайд по контрибуции](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) , если хочешь присоединиться.