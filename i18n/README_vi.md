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
    <i>CLI chính thức và thư viện Python để làm việc với Hugging Face Hub.</i>
</p>

<p align="center">
    <a href="https://huggingface.co/docs/huggingface_hub/vi/index">Tài liệu tiếng Việt</a>
    ·
    <a href="https://huggingface.co/docs/huggingface_hub/vi/installation">Cài đặt</a>
    ·
    <a href="https://huggingface.co/docs/huggingface_hub/en/guides/cli">Hướng dẫn CLI (tiếng Anh)</a>
    ·
    <a href="https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md">Đóng góp</a>
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
        <b>Tiếng Việt</b> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_hi.md">हिंदी</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_cn.md">中文 (简体)</a> |
        <a href="https://github.com/huggingface/huggingface_hub/blob/main/i18n/README_kn.md">ಕನ್ನಡ</a>
    </p>
</h4>

## Bắt đầu nhanh

Cài đặt [CLI hf](https://huggingface.co/docs/huggingface_hub/en/guides/cli) bằng trình cài đặt độc lập:

```bash
# On macOS and Linux.
curl -LsSf https://hf.co/cli/install.sh | bash
```

```powershell
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Đăng nhập, sau đó bắt đầu làm việc với Hub:

```bash
# Log in (use --token $HF_TOKEN in non-interactive environments)
hf auth login

# Find models served by Inference Providers
hf models ls --warm

# Download a model
hf download Qwen/Qwen3-0.6B

# Upload files to your own repo
hf upload username/my-cool-model ./model.safetensors

# Sync a local folder to a storage bucket
hf buckets sync ./checkpoints hf://buckets/username/my-bucket

# Run a job on Hugging Face infrastructure
hf jobs run python:3.12 python -c "print('Hello from the cloud!')"

# Discover everything else
hf --help
```

Hub dùng token để xác thực ứng dụng (xem [tài liệu về token](https://huggingface.co/docs/hub/security-tokens)). Để tìm hiểu các chức năng chính, hãy xem [hướng dẫn CLI (tiếng Anh)](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## huggingface_hub là gì?

Thư viện `huggingface_hub` giúp bạn làm việc với [Hugging Face Hub](https://huggingface.co/), nền tảng thúc đẩy học máy mã nguồn mở dành cho người tạo và cộng tác viên. Bạn có thể khám phá các mô hình được huấn luyện sẵn và bộ dữ liệu cho dự án, dùng thử hàng nghìn ứng dụng học máy trên Hub hoặc tạo và chia sẻ mô hình, bộ dữ liệu và bản demo của riêng mình với cộng đồng. Thư viện cung cấp hai giao diện trong cùng một gói: [CLI hf](https://huggingface.co/docs/huggingface_hub/en/guides/cli) cho terminal và thư viện Python `huggingface_hub`; cả hai đều được thiết kế để hỗ trợ con người lẫn AI agent. Bạn có thể dùng chúng để:

- [Tải tệp](https://huggingface.co/docs/huggingface_hub/en/guides/download) từ Hub.
- [Tải tệp lên](https://huggingface.co/docs/huggingface_hub/en/guides/upload) Hub.
- [Quản lý repo](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
- [Chạy suy luận](https://huggingface.co/docs/huggingface_hub/en/guides/inference) với mô hình đã triển khai.
- [Chạy job](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) trên hạ tầng Hugging Face.
- [Tìm kiếm](https://huggingface.co/docs/huggingface_hub/en/guides/search) mô hình, bộ dữ liệu và Spaces.
- [Chia sẻ Model Card](https://huggingface.co/docs/huggingface_hub/en/guides/model-cards) để mô tả mô hình.
- [Tham gia cộng đồng](https://huggingface.co/docs/huggingface_hub/en/guides/community) qua pull request và bình luận.
- Thực hiện tất cả các việc trên từ terminal bằng [CLI hf](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## Dành cho con người và AI agent

CLI hf được thiết kế cho cả người dùng lẫn coding agent: đầu ra của cùng một lệnh sẽ thích ứng khi agent chạy lệnh đó. Nếu bạn dùng Claude Code, Codex, Cursor hoặc một coding agent khác, hãy cài Skill của CLI hf — tài liệu tham chiếu lệnh được tạo từ CLI đã cài trên máy:

```bash
# works with Claude Code, Codex, Cursor, OpenCode, Pi and any agent that loads skills from `.agents/skills`
hf skills add
```

Tìm hiểu thêm trong [hướng dẫn Hugging Face CLI cho AI agent](https://huggingface.co/docs/hub/agents-cli) và [bài viết giới thiệu](https://huggingface.co/blog/hf-cli-for-agents).

## Dùng thư viện Python

Cài đặt gói `huggingface_hub` bằng [pip](https://pypi.org/project/huggingface-hub/) (thao tác này cũng cài CLI hf):

```bash
pip install huggingface_hub
```

Chúng tôi khuyến nghị dùng [uv](https://docs.astral.sh/uv/) để cài đặt nhanh và đáng tin cậy:

```bash
uv pip install huggingface_hub
```

Để giữ gói mặc định gọn nhẹ, `huggingface_hub` cung cấp một số dependency tùy chọn cho từng nhu cầu. Ví dụ, để dùng mô-đun MCP:

```bash
pip install "huggingface_hub[mcp]"
```

Để biết thêm về cài đặt và dependency tùy chọn, xem [hướng dẫn cài đặt](https://huggingface.co/docs/huggingface_hub/vi/installation).

### Tải tệp

Tải một tệp:

```py
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="zai-org/GLM-5.2", filename="config.json")
```

Hoặc tải toàn bộ repo:

```py
from huggingface_hub import snapshot_download

snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
```

Tệp được tải vào thư mục cache cục bộ. Xem [hướng dẫn quản lý cache (tiếng Anh)](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) để biết thêm chi tiết.

### Tạo repo

```py
from huggingface_hub import create_repo

create_repo(repo_id="super-cool-model")
```

### Tải tệp lên

Tải một tệp:

```py
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/home/lysandre/dummy-test/README.md",
    path_in_repo="README.md",
    repo_id="lysandre/test-model",
)
```

Hoặc tải cả thư mục:

```py
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/path/to/local/space",
    repo_id="username/my-cool-space",
    repo_type="space",
)
```

Xem [hướng dẫn tải tệp lên (tiếng Anh)](https://huggingface.co/docs/huggingface_hub/en/guides/upload) để biết thêm chi tiết.

## Tích hợp với Hub

Chúng tôi hợp tác với các thư viện học máy mã nguồn mở để cung cấp dịch vụ lưu trữ và quản lý phiên bản mô hình miễn phí. Xem các [tích hợp hiện có](https://huggingface.co/docs/hub/libraries).

Các lợi ích gồm:

- Lưu trữ mô hình hoặc bộ dữ liệu miễn phí cho thư viện và người dùng.
- Quản lý phiên bản tệp tích hợp, kể cả với tệp rất lớn, nhờ [Xet](https://huggingface.co/docs/hub/xet/index) — hệ thống lưu trữ chia khối và loại bỏ dữ liệu trùng lặp của Hub.
- Widget chạy trực tiếp mô hình đã tải lên ngay trong trình duyệt.
- Bất kỳ ai cũng có thể tải mô hình mới cho thư viện; chỉ cần thêm tag tương ứng để mô hình có thể được tìm thấy.
- Tốc độ tải nhanh: CloudFront (CDN) nhân bản dữ liệu theo khu vực để tăng tốc truy cập trên toàn cầu.
- Thống kê mức sử dụng và nhiều tính năng khác trong tương lai.

Nếu muốn tích hợp thư viện của mình, bạn có thể mở issue để bắt đầu thảo luận. Chúng tôi đã viết [hướng dẫn từng bước](https://huggingface.co/docs/hub/adding-a-library) để hỗ trợ quá trình tích hợp.

## Hoan nghênh đóng góp (đề xuất tính năng, báo lỗi, v.v.) 💙💚💛💜🧡❤️

Mọi người đều có thể đóng góp và mọi đóng góp đều được trân trọng. Không chỉ viết mã mới giúp ích cho cộng đồng: trả lời câu hỏi, hỗ trợ người khác và cải thiện tài liệu cũng rất có giá trị. Hãy xem [hướng dẫn đóng góp](https://github.com/huggingface/huggingface_hub/blob/main/CONTRIBUTING.md) để biết cách bắt đầu đóng góp cho repo này.
