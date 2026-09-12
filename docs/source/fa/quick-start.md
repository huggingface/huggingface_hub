<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# شروع سریع

[Hugging Face Hub](https://huggingface.co/) مرجع اصلی اشتراک‌گذاری مدل‌های یادگیری ماشین، دموها، دیتاست‌ها و معیارهای ارزیابی است. کتابخانه‌ی `huggingface_hub` به شما کمک می‌کند بدون خارج‌شدن از محیط توسعه‌تان با Hub کار کنید. با آن می‌توانید به‌سادگی مخزن بسازید و مدیریت کنید، فایل‌ها را دانلود یا آپلود کنید و فراداده‌های مفید مدل و دیتاست را از Hub بگیرید.

## نصب

برای شروع، کتابخانه‌ی `huggingface_hub` را نصب کنید:

```bash
pip install --upgrade huggingface_hub
```

برای جزئیات بیشتر، راهنمای [نصب](installation) را ببینید.

> [!TIP]
> `huggingface_hub` یک CLI با نام [`hf`](./guides/cli) هم دارد که از طریق آن می‌توانید مستقیماً از ترمینال با Hub کار کنید.
> اگر از agentهای هوش مصنوعی مانند Claude Code، Codex یا Cursor استفاده می‌کنید، Skill را نصب کنید تا agent شما بتواند از CLI استفاده کند:
> ```bash
> # for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
> hf skills add
> # includes the above + Claude Code
> hf skills add --claude
> ```
> برای جزئیات بیشتر، راهنمای [Hugging Face CLI for AI Agents](https://huggingface.co/docs/hub/agents-cli) را بخوانید.

## دانلود فایل‌ها

مخزن‌های Hub با git version-control می‌شوند و می‌توانید یک فایل یا کل مخزن را دانلود کنید. برای دانلود فایل‌ها از تابع [`hf_hub_download`] استفاده کنید. این تابع فایل را دانلود و روی دیسک محلی cache می‌کند؛ بنابراین در دفعات بعدی همان فایل از cache خوانده می‌شود و نیازی به دانلود دوباره نیست.

به شناسه‌ی مخزن و نام فایلی که می‌خواهید دانلود کنید نیاز دارید. مثلاً برای دانلود فایل پیکربندی مدل [Pegasus](https://huggingface.co/google/pegasus-xsum):

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

برای دانلود یک نسخه‌ی مشخص از فایل، با پارامتر `revision` نام branch، tag یا commit hash را تعیین کنید. اگر از commit hash استفاده می‌کنید، باید hash کامل را وارد کنید؛ hash کوتاه هفت‌کاراکتری پذیرفته نمی‌شود:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

برای جزئیات و گزینه‌های بیشتر، مرجع API مربوط به [`hf_hub_download`] را ببینید.

<a id="login"></a> <!-- backward compatible anchor -->

## احراز هویت

در بسیاری از کارها برای تعامل با Hub باید با حساب Hugging Face خود احراز هویت شوید؛ مثلاً دانلود مخزن خصوصی، آپلود فایل یا ساخت PR. اگر حساب ندارید، ابتدا [یک حساب بسازید](https://huggingface.co/join).

### دستور ورود

ساده‌ترین روش احراز هویت، استفاده از دستور [`login`] است:

```bash
hf auth login
```

اگر از قبل وارد شده باشید، دستور بلافاصله تمام می‌شود. برای ورود دوباره از `hf auth login --force` استفاده کنید. اگر وارد نشده باشید، از شما خواسته می‌شود در مرورگر وارد شوید: URL چاپ‌شده را باز کنید، کد کوتاه را وارد و درخواست را تأیید کنید. سپس access token دریافت و در مسیر `HF_HOME` شما ذخیره می‌شود که به‌صورت پیش‌فرض `~/.cache/huggingface/token` است. token پس از مدتی منقضی می‌شود، اما تا وقتی از آن استفاده می‌کنید به‌طور خودکار refresh می‌شود. هر اسکریپت یا کتابخانه‌ای که با Hub کار کند، هنگام ارسال درخواست از این token استفاده می‌کند. همچنین می‌توانید [User Access Token](https://huggingface.co/docs/hub/security-tokens) ساخته‌شده در [صفحه‌ی Settings](https://huggingface.co/settings/tokens) را paste کنید.

> [!TIP]
> User Access Token می‌تواند permission `read` یا `write` داشته باشد. اگر می‌خواهید مخزن بسازید یا ویرایش کنید، از token با دسترسی `write` استفاده کنید. در غیر این صورت بهتر است برای کاهش ریسک در صورت افشای ناخواسته‌ی token، یک token با دسترسی `read` بسازید.

همچنین می‌توانید در notebook یا script با [`login`] به‌صورت برنامه‌نویسی‌شده وارد شوید:

```py
>>> from huggingface_hub import login
>>> login()
```

در هر لحظه فقط می‌توانید با یک حساب وارد شده باشید. ورود با حساب جدید، شما را به‌طور خودکار از حساب قبلی خارج می‌کند. برای دیدن حساب فعال، دستور `hf auth whoami` را اجرا کنید.

> [!WARNING]
> پس از ورود، همه‌ی درخواست‌ها به Hub، حتی متدهایی که لزوماً به احراز هویت نیاز ندارند، به‌طور پیش‌فرض از access token شما استفاده می‌کنند. اگر می‌خواهید استفاده‌ی ضمنی از token را غیرفعال کنید، متغیر محیطی `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` را تنظیم کنید. [مرجع](../package_reference/environment_variables#hfhubdisableimplicittoken) را ببینید.

### مدیریت چند token در سیستم محلی

با اجرای [`login`] برای هر token می‌توانید چند token را روی سیستم خود ذخیره کنید. اگر لازم است بین tokenهای محلی جابه‌جا شوید، از دستور [`auth switch`] استفاده کنید:

```bash
hf auth switch
```

این دستور فهرستی از tokenهای ذخیره‌شده را با نامشان نشان می‌دهد تا یکی را انتخاب کنید. پس از انتخاب، token انتخاب‌شده به token _active_ تبدیل می‌شود و برای تمام تعامل‌ها با Hub استفاده خواهد شد.

برای دیدن همه‌ی access tokenهای موجود روی دستگاه، از `hf auth list` استفاده کنید.

### متغیر محیطی

برای احراز هویت می‌توانید از متغیر محیطی `HF_TOKEN` هم استفاده کنید. این روش به‌خصوص در Spaceهایی مفید است که می‌توانید `HF_TOKEN` را به‌عنوان [Space secret](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) تنظیم کنید.

> [!TIP]
> **جدید:** Google Colaboratory به شما اجازه می‌دهد برای notebookها [private key](https://twitter.com/GoogleColab/status/1719798406195867814) تعریف کنید. یک secret با نام `HF_TOKEN` بسازید تا به‌طور خودکار احراز هویت شوید.

احراز هویت از طریق متغیر محیطی یا secret بر token ذخیره‌شده در دستگاه اولویت دارد.

### پارامترهای متد

در آخر، می‌توانید token را به هر متدی که پارامتر `token` می‌پذیرد ارسال کنید:

```
from huggingface_hub import whoami

user = whoami(token=...)
```

این روش معمولاً توصیه نمی‌شود، مگر در محیطی که نمی‌خواهید token را دائمی ذخیره کنید یا لازم است هم‌زمان چند token را مدیریت کنید.

> [!WARNING]
> هنگام ارسال token به‌عنوان پارامتر احتیاط کنید. بهترین روش این است که token را از یک vault امن بخوانید و آن را در codebase یا notebook hardcode نکنید. در صورت اشتراک‌گذاری ناخواسته‌ی کد، tokenهای hardcode‌شده خطر افشا دارند.

## ساخت مخزن

بعد از ثبت‌نام و ورود، با تابع [`create_repo`] یک مخزن بسازید:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

اگر می‌خواهید مخزن private باشد:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

مخزن‌های private جز برای شما برای کسی قابل مشاهده نیستند.

> [!TIP]
> برای ساخت مخزن یا push کردن محتوا به Hub باید User Access Token با permission `write` داشته باشید. هنگام ساخت token در [صفحه‌ی Settings](https://huggingface.co/settings/tokens) می‌توانید permission آن را انتخاب کنید.

## آپلود فایل‌ها

برای اضافه‌کردن فایل به مخزنی که تازه ساخته‌اید، از تابع [`upload_file`] استفاده کنید. باید این موارد را تعیین کنید:

1. مسیر فایل برای آپلود.
2. مسیر فایل در مخزن.
3. شناسه‌ی مخزنی که فایل را به آن اضافه می‌کنید.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

برای آپلود هم‌زمان بیش از یک فایل، راهنمای [Upload](./guides/upload) را ببینید. این راهنما چند روش آپلود، با git یا بدون آن، را معرفی می‌کند.

## گام‌های بعدی

کتابخانه‌ی `huggingface_hub` راهی ساده برای تعامل با Hub از طریق Python فراهم می‌کند. برای یادگیری بیشتر درباره‌ی مدیریت فایل‌ها و مخزن‌ها در Hub، پیشنهاد می‌کنیم [how-to guideها](./guides/overview) را بخوانید تا بتوانید:

- [مخزن خود را مدیریت کنید](./guides/repository).
- فایل‌ها را از Hub [دانلود](./guides/download) کنید.
- فایل‌ها را به Hub [آپلود](./guides/upload) کنید.
- برای یافتن مدل یا دیتاست موردنظر خود در Hub [جست‌وجو](./guides/search) کنید.
- برای مدل‌های میزبانی‌شده روی Hugging Face Hub، در چند service مختلف [Inference](./guides/inference) اجرا کنید.
