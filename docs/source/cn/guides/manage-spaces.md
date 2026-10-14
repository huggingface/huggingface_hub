<!--⚠️ 本文件采用 Markdown 格式，但包含文档构建工具 doc-builder 的专用语法（类似 MDX），
因此在普通 Markdown 查看器中可能无法正确显示。
-->

# 管理你的 Space

本指南介绍如何使用 `huggingface_hub` 管理 Space 的运行配置，包括[密钥](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)、[硬件](https://huggingface.co/docs/hub/spaces-gpus)和存储卷。

## 搜索 Space

你可以使用 [`search_spaces`] 在 Hub 上进行语义搜索，查找 Space。对于包含多个词的查询，该方法使用基于嵌入向量的搜索；对于单个词的查询，则使用全文搜索。

```py
>>> from huggingface_hub import search_spaces
>>> results = list(search_spaces("generate image"))
>>> results[0]
SpaceSearchResult(id='mrfakename/Z-Image-Turbo', title='Z Image Turbo', sdk='gradio', likes=2867, ...)
```

你可以按 SDK 或标签筛选结果：

```py
>>> results = search_spaces("chatbot", sdk="gradio", filter="mcp-server")
```

也可以通过命令行（CLI）搜索：

```bash
>>> hf spaces search "generate image"
>>> hf spaces search "chatbot" --sdk gradio --limit 5
```

## 简单示例：配置密钥和硬件

下面通过一个完整示例，介绍如何在 Hub 上创建和配置 Space。

### 在 Hub 上创建 Space

```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# For example with a Gradio SDK
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

> [!TIP]
> 静态 Space 对所有人免费。Gradio 和 Docker Space 需要计算资源：使用免费的 `cpu-basic` 硬件时，需要订阅相应套餐（个人用户为 PRO，组织为 Team 或 Enterprise）；使用付费硬件则只需要配置计费方式（支付方式和预付额度）。如果没有订阅，创建、复制或迁移使用 `cpu-basic` 的 Gradio 或 Docker Space 会因支付错误（HTTP 402）而失败。详情请参阅[价格说明](https://huggingface.co/pricing)。

### 从模板创建 Space

你可以使用 Hub 提供的官方模板（例如 JupyterLab、Gradio 聊天机器人或 Streamlit 应用）来初始化新 Space，无需从空白开始。先用 [`list_space_templates`] 列出可用模板，再将模板的 `repo_id`（或简短的 `name`）作为 `space_template` 参数传给 [`create_repo`]。请注意，仍然需要提供 `space_sdk`：模板用于初始化文件，而 SDK 用于设置卡片元数据。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# List available templates
>>> for template in api.list_space_templates():
...     print(f"{template.name} ({template.repo_id})")
Streamlit (streamlit/streamlit-template-space)
JupyterLab (SpacesExamples/jupyterlab)
chatbot (gradio-templates/chatbot)
...

# Create a Space from a template, by repo id or by short name
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_template="SpacesExamples/jupyterlab")
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_template="JupyterLab")
```

某些模板（例如 JupyterLab）建议设为私有。如果你没有显式设置可见性，使用这类模板创建的 Space 会自动设为私有。

命令行中的 `hf spaces templates` 和 `hf repos create --template` 提供了相同功能：

```bash
# List available templates
>>> hf spaces templates
NAME            REPO_ID                                        SDK    PREFERRED_PRIVATE
--------------- ---------------------------------------------- ------ -----------------
Streamlit       streamlit/streamlit-template-space             docker
JupyterLab      SpacesExamples/jupyterlab                      docker ✔
Argilla         argilla/argilla-template-space                 docker
Livebook        livebook-dev/livebook                          docker
...

# Create a Space from a template
>>> hf repos create my-jupyterlab --type space --template jupyterlab
```

### 复制 Space

如果你希望基于已有 Space 继续开发，而不是从头开始，可以复制它。如果你想自行控制某个公开 Space 的配置，也可以采用这种方式。详情请参阅 [`duplicate_repo`]。

```py
>>> api.duplicate_repo("multimodalart/dreambooth-training", repo_type="space")
```

### 使用你喜欢的方式上传代码

下面的示例将本机的 `src/` 文件夹上传到你的 Space：

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

到这一步，你的应用应该已经在 Hub 上运行了！接下来，你可能还需要配置密钥或升级硬件。

### 配置密钥和变量

你的 Space 可能需要密钥、令牌或变量才能运行。例如，Space 生成图像后，需要使用 HF 令牌将图像数据集上传到 Hub。详情请参阅[文档](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)。

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

你可以列出现有的密钥和变量。密钥的值只能写入，不能读取，因此返回结果仅包含键名、描述和更新时间戳：

```py
>>> api.get_space_secrets(repo_id=repo_id)
{'HF_TOKEN': SpaceSecret(key='HF_TOKEN', description=None, updated_at=datetime.datetime(...))}
>>> api.get_space_variables(repo_id=repo_id)
{'MODEL_REPO_ID': SpaceVariable(key='MODEL_REPO_ID', value='user/repo', description=None, updated_at=...)}
```

你也可以删除密钥和变量：

```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```

> [!TIP]
> 在 Space 内部，可以直接通过环境变量读取密钥（如果使用 Streamlit，也可以通过其密钥管理功能读取），无需调用 API 获取！

> [!WARNING]
> 更改 Space 的配置（密钥或硬件）会触发应用重启。

**补充：在创建或复制 Space 时设置密钥和变量！**

你可以在创建或复制 Space 时直接设置密钥和变量：

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_secrets=[{"key": "HF_TOKEN", "value": "hf_api_***"}, ...],
...     space_variables=[{"key": "MODEL_REPO_ID", "value": "user/repo"}, ...],
... )
```

```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_secrets=[{"key": "HF_TOKEN", "value": "hf_api_***"}, ...],
...     space_variables=[{"key": "MODEL_REPO_ID", "value": "user/repo"}, ...],
... )
```

### 配置硬件

默认情况下，Space 使用免费的 `cpu-basic` 硬件运行（如上所述，Gradio 和 Docker Space 需要订阅套餐）。你可以升级硬件，让 Space 在 GPU 上运行。升级需要支付方式和预付额度，或社区资助。详情请参阅[文档](https://huggingface.co/docs/hub/spaces-gpus)。

```py
# Use `SpaceHardware` enum
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# Or simply pass a string value
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

硬件升级不会立即完成，因为服务器需要重新加载你的 Space。你可以随时检查 Space 当前使用的硬件，确认升级请求是否已生效。

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

现在，你的 Space 已配置完成。使用结束后，记得将硬件降回 `cpu-basic`。

**补充：在创建或复制 Space 时申请硬件！**

Space 构建完成后，会自动分配升级后的硬件。

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_hardware="cpu-upgrade",
...     space_sleep_time="7200", # 2 hours in secs
... )
```
```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_hardware="cpu-upgrade",
...     space_sleep_time="7200", # 2 hours in secs
... )
```

### 查看 ZeroGPU 配额

[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) Space 使用按需分配的共享 GPU。每次通过界面或 API 使用 ZeroGPU Space，都会从你的个人配额中扣除一定的 GPU 使用时间，配额会定期重置。如果你正在基于 ZeroGPU Space 构建应用、智能体或 MCP 服务器，可以使用 [`get_zero_gpu_quota`] 跟踪用量。返回值以 GPU 秒为单位。

```py
>>> quota = api.get_zero_gpu_quota()
>>> quota
ZeroGpuQuota(base=2400, remaining=1810, resets_at=datetime.datetime(2026, 9, 30, 9, 12, 3, tzinfo=datetime.timezone.utc), overquota_used=0)
>>> if quota.remaining < 60:
...     print(f"Less than 1 minute of ZeroGPU left. Quota resets at {quota.resets_at}.")
```

如果自上次配额重置以来你尚未使用 ZeroGPU，`resets_at` 为 `None`。如果使用细粒度令牌，它必须具有“Billing > Read billing usage and payment method status”（计费 > 读取计费用量和支付方式状态）权限。你也可以通过命令行 `hf spaces zero-gpu quota` 获取相同信息。

配额用尽后，PRO、Team 和 Enterprise 用户可以消耗预付额度来继续使用 ZeroGPU Space。你可以在[计费设置](https://huggingface.co/settings/billing?add-credits=true)中购买额度。详情请参阅[使用额度扩展配额](https://huggingface.co/docs/hub/spaces-zerogpu#extending-quota-with-credits)。

### 暂停和重启 Space

默认情况下，使用升级硬件的 Space 会持续运行。为了避免在不使用时继续计费，你可以通过 [`pause_space`] 暂停它。暂停后的 Space 会保持停用，直到所有者通过界面或 [`restart_space`] API 将其重启。关于暂停模式的更多说明，请参阅[这一节](https://huggingface.co/docs/hub/spaces-gpus#pause)。

```py
# Pause your Space to avoid getting billed
>>> api.pause_space(repo_id=repo_id)
# (...)
# Restart it when you need it
>>> api.restart_space(repo_id=repo_id)
```

另一种方式是设置休眠超时时间。如果 Space 在指定时间内没有活动，就会进入休眠。之后任何访客访问该 Space 都会将它唤醒。你可以使用 [`set_space_sleep_time`] 设置超时时间。关于休眠模式的更多说明，请参阅[这一节](https://huggingface.co/docs/hub/spaces-gpus#sleep-time)。

```py
# Put your Space to sleep after 1h of inactivity
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

注意：使用 `cpu-basic` 硬件时，无法自定义休眠时间。Space 会在连续 48 小时没有活动后自动暂停。

**补充：申请硬件时设置休眠时间**

Space 构建完成后，会自动分配升级后的硬件。

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**补充：创建或复制 Space 时设置休眠时间！**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```
```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```

### 通过日志排查 Space 故障

当 Space 构建失败或在运行时崩溃时，除了在浏览器中查看日志，你还可以通过 [`fetch_space_logs`] 在程序中读取这些日志。对于无法打开浏览器的脚本或智能体工作流，这种方式尤其有用。

```py
# Drain the currently available run logs and return immediately (like `docker logs`)
>>> for line in api.fetch_space_logs(repo_id=repo_id):
...     print(line, end="")

# Read the container build logs instead (useful when the Space is stuck in BUILD_ERROR)
>>> for line in api.fetch_space_logs(repo_id=repo_id, build=True):
...     print(line, end="")

# Stream run logs in real time until the server closes the stream (Ctrl-C to stop)
>>> for line in api.fetch_space_logs(repo_id=repo_id, follow=True):
...     print(line, end="")
```

命令行也提供了相同功能：

```bash
hf spaces logs username/my-space             # drain run logs
hf spaces logs username/my-space --build     # read build logs
hf spaces logs username/my-space -f          # stream in real time
hf spaces logs username/my-space -n 50       # last 50 lines only
```

### 通过 SSH 连接 Space（开发模式）

[开发模式（Dev Mode）](https://huggingface.co/docs/hub/spaces-dev-mode)允许你通过 SSH 连接正在运行的 Space 容器，进行实时调试和开发。使用 `hf spaces ssh` 即可直接从终端建立会话。如果 Space 尚未启用开发模式，命令行会提示你启用它（也可以传入 `--auto` 跳过提示）。

你必须先在[设置](https://huggingface.co/settings/keys)中注册 SSH 公钥。

```bash
# SSH into a Space (enables Dev Mode if needed)
hf spaces ssh username/my-space

# Auto-enable Dev Mode without prompting
hf spaces ssh username/my-space --auto

# Print the SSH command without running it
hf spaces ssh username/my-space --dry-run

# Use a specific SSH key
hf spaces ssh username/my-space -i ~/.ssh/id_ed25519
```

你也可以使用 `hf spaces dev-mode` 只启用开发模式，而不建立 SSH 连接。该命令会显示 SSH、VS Code、Cursor 和 Windsurf 的连接说明：

```bash
hf spaces dev-mode username/my-space
```

启用开发模式需要 PRO、Team 或 Enterprise 套餐。关闭开发模式则始终允许，例如在开发模式开启期间订阅已到期的情况下：

```bash
hf spaces dev-mode username/my-space --stop
```

### 在 Space 中挂载存储卷

你可以将 Hub 上的资源（模型、数据集或存储桶）作为存储卷挂载到 Space 容器中。这样，Space 就能通过文件系统直接访问这些资源，无需在代码中下载。你可以在创建或复制 Space 时直接配置存储卷：

```py
>>> from huggingface_hub import Volume
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/data"),
...     ],
... )
```
```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/data"),
...     ],
... )
```

你可以通过 Space 的运行时信息查看当前挂载的存储卷：

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.volumes
[Volume(type='model', source='username/my-model', mount_path='/models', read_only=True), ...]
```

如果需要更新已有 Space 的存储卷，请使用 [`set_space_volumes`]。注意，这会替换此前挂载的全部存储卷。

```py
>>> api.set_space_volumes(
...     repo_id=repo_id,
...     volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="dataset", source="username/my-dataset", mount_path="/data", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/output"),
...     ],
... )
```

移除 Space 中的全部存储卷：

```py
>>> api.delete_space_volumes(repo_id=repo_id)
```

> [!NOTE]
> 模型、数据集和 Space 始终以只读方式挂载。只有存储桶支持读写挂载。

> [!WARNING]
> 设置存储卷会替换此前挂载的全部存储卷。如果只想新增一个存储卷，请先从运行时信息中读取现有列表，并将其包含在新的列表中。

所有存储卷操作都可以通过命令行完成：

```bash
# List current volumes
hf spaces volumes ls username/my-space

# Set (replace) volumes
hf spaces volumes set username/my-space \
    -v hf://models/username/my-model:/models \
    -v hf://buckets/username/my-bucket:/data

# Remove all volumes
hf spaces volumes delete username/my-space
```

## 进阶用法：临时升级 Space

Space 支持多种使用场景。有时，你可能只想临时使用特定硬件运行 Space，完成任务后再关闭。本节介绍如何利用 Space 按需微调模型。这只是解决该问题的一种思路，你可以根据自己的使用场景进行调整。

假设我们有一个用于微调模型的 Space。它是一个 Gradio 应用，接收模型 ID 和数据集 ID 作为输入，工作流程如下：

0. 提示用户选择模型和数据集。
1. 从 Hub 加载模型。
2. 从 Hub 加载数据集。
3. 使用该数据集微调模型。
4. 将新模型上传到 Hub。

第 3 步需要特定硬件，但你并不希望 Space 一直在付费 GPU 上运行。一种解决方式是：开始训练时动态申请硬件，训练结束后再停止使用。由于申请硬件会重启 Space，应用需要以某种方式“记住”当前正在执行的任务。实现这一点有多种方法，本指南会介绍一种使用数据集作为“任务调度器”的方案。

### 应用框架

应用的大致结构如下：启动时检查是否有待执行任务；如果有，就在相应硬件上执行。完成后，将硬件切回免费的 `cpu-basic` CPU，并提示用户提交新任务。

> [!WARNING]
> 这种工作流程不支持普通演示应用那样的并发访问。尤其是在训练期间，界面将不可用。建议将仓库设为私有，确保只有你自己使用。

```py
# Space will need your token to request hardware: set it as a Secret !
HF_TOKEN = os.environ.get("HF_TOKEN")

# Space own repo_id
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# On Space startup, check if a task is scheduled. If yes, finetune the model. If not,
# display an interface to request a new task.
task = get_task()
if task is None:
    # Start Gradio app
    def gradio_fn(task):
        # On user request, add task and request hardware
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # Check if Space is loaded with a GPU.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # If yes, finetune base model on dataset !
        train_and_upload(task)

        # Then, mark the task as "DONE"
        mark_as_done(task)

        # DO NOT FORGET: set back CPU hardware
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### 任务调度器

调度任务的方法有很多。下面的示例使用保存在数据集仓库中的一个简单 CSV 文件来实现。

```py
# Dataset ID in which a `tasks.csv` file contains the tasks to perform.
# Here is a basic example for `tasks.csv` containing inputs (base model and dataset)
# and status (PENDING or DONE).
#     multimodalart/sd-fine-tunable,Wauplin/concept-1,DONE
#     multimodalart/sd-fine-tunable,Wauplin/concept-2,PENDING
TASK_DATASET_ID = "Wauplin/dreambooth-task-scheduler"

def _get_csv_file():
    return hf_hub_download(repo_id=TASK_DATASET_ID, filename="tasks.csv", repo_type="dataset", token=HF_TOKEN)

def get_task():
    with open(_get_csv_file()) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] == "PENDING":
                return row[0], row[1] # model_id, dataset_id

def add_task(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Quick and dirty way to add a task
        path_or_fileobj=(tasks + f"\n{model_id},{dataset_id},PENDING").encode()
    )

def mark_as_done(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Quick and dirty way to set the task as DONE
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```
