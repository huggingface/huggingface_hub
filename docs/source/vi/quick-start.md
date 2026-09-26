<!--⚠️ Tệp này dùng Markdown cùng một số cú pháp riêng của doc-builder (tương tự MDX), nên có thể không hiển thị đúng trong một số trình xem Markdown.
-->

# Bắt đầu nhanh

[Hugging Face Hub](https://huggingface.co/) là nơi chia sẻ các mô hình học máy, bản demo, bộ dữ liệu và chỉ số đánh giá. Thư viện huggingface_hub giúp bạn làm việc với Hub ngay trong môi trường phát triển của mình. Bạn có thể dễ dàng tạo và quản lý repo, tải tệp lên hoặc tải tệp xuống, đồng thời lấy thông tin hữu ích về mô hình và bộ dữ liệu.

## Cài đặt

Để bắt đầu, hãy cài đặt thư viện huggingface_hub:

```bash
pip install --upgrade huggingface_hub
```

Xem thêm chi tiết trong [hướng dẫn cài đặt](installation).

> [!TIP]
> huggingface_hub cũng đi kèm [CLI hf](https://huggingface.co/docs/huggingface_hub/en/guides/cli), cho phép bạn tương tác trực tiếp với Hub từ terminal.
> Nếu dùng AI agent (Claude Code, Codex, Cursor, ...), hãy cài Skill để agent có thể dùng CLI:
> ```bash
> # works with Claude Code, Codex, Cursor, OpenCode, Pi and any agent that loads skills from `.agents/skills`
> hf skills add
> ```
> Xem thêm trong [hướng dẫn Hugging Face CLI cho AI agent](https://huggingface.co/docs/hub/agents-cli).

## Tải tệp

Các repo trên Hub được quản lý phiên bản bằng Git. Bạn có thể tải một tệp riêng lẻ hoặc toàn bộ repo bằng hàm hf_hub_download. Hàm này tải tệp về máy và lưu vào cache cục bộ. Lần sau cần dùng tệp đó, thư viện sẽ đọc từ cache nên không phải tải lại.

Bạn cần biết ID của repo và tên tệp muốn tải. Ví dụ, để tải tệp cấu hình của mô hình [Pegasus](https://huggingface.co/google/pegasus-xsum):

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

Để tải một phiên bản cụ thể của tệp, dùng tham số revision để chỉ định tên nhánh, tag hoặc mã commit. Nếu chọn mã commit, bạn phải dùng đầy đủ mã hash, không dùng dạng rút gọn 7 ký tự:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

Xem thêm các tùy chọn trong tài liệu tham chiếu API của hf_hub_download.

<a id="login"></a> <!-- backward compatible anchor -->

## Xác thực

Trong nhiều trường hợp, bạn cần đăng nhập để làm việc với Hub, chẳng hạn khi tải repo riêng tư, tải tệp lên hoặc tạo pull request. Nếu chưa có tài khoản Hugging Face, hãy [tạo tài khoản](https://huggingface.co/join).

### Lệnh đăng nhập

Cách đơn giản nhất để xác thực là dùng lệnh login:

```bash
hf auth login
```

Nếu bạn đã đăng nhập, lệnh sẽ kết thúc ngay. Nếu chưa, bạn sẽ được yêu cầu xác thực qua trình duyệt: mở URL được in ra, nhập mã ngắn, chấp thuận yêu cầu; sau đó token truy cập sẽ được lấy và lưu trong thư mục HF_HOME (mặc định là ~/.cache/huggingface/token). Token sẽ hết hạn sau một thời gian nhưng được tự động làm mới nếu bạn tiếp tục sử dụng. Mọi script hoặc thư viện tương tác với Hub đều dùng token này khi gửi yêu cầu. Ngoài ra, bạn có thể dán [User Access Token](https://huggingface.co/docs/hub/security-tokens) được tạo từ trang [Settings](https://huggingface.co/settings/tokens).

> [!TIP]
> User Access Token có thể được cấp quyền read hoặc write. Khi cần tạo hoặc chỉnh sửa repo, hãy dùng token có quyền write. Nếu không, nên tạo token chỉ có quyền read để giảm rủi ro khi token vô tình bị lộ.

Bạn cũng có thể đăng nhập bằng cách gọi login trong notebook hoặc script:

```py
>>> from huggingface_hub import login
>>> login()
```

Một thời điểm bạn chỉ có thể đăng nhập vào một tài khoản. Đăng nhập tài khoản mới sẽ tự động đăng xuất tài khoản trước đó. Để kiểm tra tài khoản hiện đang dùng, chạy lệnh hf auth whoami.

> [!WARNING]
> Sau khi đăng nhập, mọi yêu cầu gửi tới Hub — kể cả các phương thức không nhất thiết cần xác thực — mặc định đều sử dụng token truy cập của bạn. Để tắt việc tự động dùng token ngầm định, đặt biến môi trường HF_HUB_DISABLE_IMPLICIT_TOKEN=1 (xem [tài liệu tham chiếu](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubdisableimplicittoken)).

### Quản lý nhiều token trên máy

Bạn có thể lưu nhiều token bằng cách đăng nhập lần lượt bằng lệnh login. Khi cần chuyển token đang hoạt động, dùng lệnh auth switch:

```bash
hf auth switch
```

Lệnh sẽ hiển thị danh sách token đã lưu để bạn chọn theo tên. Token được chọn sẽ trở thành token đang hoạt động và được dùng cho các thao tác tiếp theo.

Để liệt kê các token truy cập hiện có trên máy, hãy chạy lệnh hf auth list.

### Biến môi trường

Bạn cũng có thể dùng biến môi trường HF_TOKEN để xác thực. Cách này đặc biệt hữu ích trong Space, nơi bạn có thể khai báo HF_TOKEN dưới dạng [secret của Space](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

Token được cung cấp qua biến môi trường hoặc secret sẽ được ưu tiên hơn token đã lưu trên máy.

> [!TIP]
> **Mới:** Google Colaboratory cho phép bạn khai báo [khóa bí mật](https://twitter.com/GoogleColab/status/1719798406195867814) riêng tư. Hãy tạo secret HF_TOKEN để được tự động xác thực!

### Tham số của phương thức

Cuối cùng, bạn có thể truyền token vào bất kỳ phương thức nào chấp nhận tham số token:

```
from huggingface_hub import whoami

user = whoami(token=...)
```

Thông thường không nên làm vậy, trừ khi bạn không muốn lưu token lâu dài trong môi trường đó. Nếu cần dùng cách này, hãy lưu token an toàn thay vì ghi trực tiếp vào notebook hoặc mã nguồn.

> [!WARNING]
> Hãy cẩn thận khi truyền token trực tiếp làm tham số. Thực hành tốt nhất là đọc token từ kho lưu trữ bí mật an toàn thay vì hard-code. Nếu vô tình chia sẻ notebook hoặc mã nguồn, token được hard-code có thể bị lộ.

## Tạo repo

Sau khi đăng ký và đăng nhập, hãy tạo repo bằng hàm create_repo:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

Để tạo repo riêng tư:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Chỉ bạn mới có thể xem repo riêng tư.

> [!TIP]
> Để tạo repo hoặc tải nội dung lên Hub, bạn cần User Access Token có quyền write. Bạn có thể chọn quyền khi tạo token trong [trang Settings](https://huggingface.co/settings/tokens).

## Tải tệp lên

Dùng hàm upload_file để thêm một tệp vào repo vừa tạo. Bạn cần chỉ định:

1. Đường dẫn tới tệp hoặc đối tượng tệp.
2. Đường dẫn sẽ dùng cho tệp trong repo.
3. ID của repo sẽ nhận tệp.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

Để tải lên nhiều tệp cùng lúc, xem [hướng dẫn Upload (tiếng Anh)](https://huggingface.co/docs/huggingface_hub/en/guides/upload) để tìm hiểu các phương thức tải lên (có hoặc không dùng Git).

## Bước tiếp theo

Thư viện huggingface_hub giúp bạn làm việc với Hub bằng Python. Để tìm hiểu thêm về cách quản lý tệp và repo, hãy xem [các hướng dẫn thực hành (tiếng Anh)](https://huggingface.co/docs/huggingface_hub/en/guides/overview), trong đó có cách:

- [Quản lý repo](https://huggingface.co/docs/huggingface_hub/en/guides/repository).
- [Tải tệp xuống](https://huggingface.co/docs/huggingface_hub/en/guides/download) từ Hub.
- [Tải tệp lên](https://huggingface.co/docs/huggingface_hub/en/guides/upload) Hub.
- [Tìm kiếm](https://huggingface.co/docs/huggingface_hub/en/guides/search) mô hình hoặc bộ dữ liệu trên Hub.
- [Chạy suy luận](https://huggingface.co/docs/huggingface_hub/en/guides/inference) qua các dịch vụ dành cho mô hình được lưu trữ trên Hugging Face Hub.
