<!--⚠️ Tệp này dùng Markdown cùng một số cú pháp riêng của doc-builder (tương tự MDX), nên có thể không hiển thị đúng trong một số trình xem Markdown.
-->

# Cài đặt

Trước khi bắt đầu, hãy chuẩn bị môi trường bằng cách cài đặt các gói cần thiết.

Thư viện huggingface_hub được kiểm thử với **Python 3.10 trở lên**.

## Cài đặt bằng pip

Chúng tôi đặc biệt khuyến nghị cài đặt huggingface_hub trong một [môi trường ảo](https://docs.python.org/3/library/venv.html). Nếu chưa quen với môi trường ảo của Python, hãy xem [hướng dẫn này](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Môi trường ảo giúp bạn quản lý từng dự án dễ hơn và tránh xung đột phiên bản dependency.

Trước tiên, hãy tạo môi trường ảo trong thư mục dự án:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo. Trên Linux và macOS:

```bash
source .venv/bin/activate
```

Trên Windows:

```bash
.venv/Scripts/activate
```

Bây giờ, bạn có thể cài đặt huggingface_hub từ [kho PyPI](https://pypi.org/project/huggingface-hub/):

```bash
pip install --upgrade huggingface_hub
```

Sau khi cài đặt, hãy [kiểm tra](#check-installation) để xác nhận mọi thứ hoạt động đúng.

### Cài đặt dependency tùy chọn

Một số dependency của huggingface_hub là [tùy chọn](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) vì chúng không cần thiết cho các chức năng cốt lõi. Tuy nhiên, một số tính năng sẽ không khả dụng nếu thiếu dependency tương ứng.

Bạn có thể cài dependency tùy chọn bằng pip:

```bash
# Install dependencies for both torch-specific and MCP-specific features.
pip install 'huggingface_hub[mcp,torch]'
```

Danh sách dependency tùy chọn của huggingface_hub:

- fastai, torch: dependency để chạy các tính năng dành riêng cho framework.
- dev: dependency phục vụ việc đóng góp cho thư viện. Gói này bao gồm testing (chạy kiểm thử), typing (kiểm tra kiểu) và quality (chạy linter).

### Cài đặt từ mã nguồn

Trong một số trường hợp, bạn có thể muốn cài trực tiếp huggingface_hub từ mã nguồn. Cách này cho phép dùng phiên bản main mới nhất thay vì bản ổn định gần nhất. Phiên bản main hữu ích khi bạn muốn cập nhật ngay sau khi một lỗi được sửa nhưng chưa có bản phát hành chính thức mới.

Tuy nhiên, phiên bản main không phải lúc nào cũng ổn định. Nhóm dự án cố gắng giữ nhánh main hoạt động tốt và thường xử lý sự cố trong vài giờ đến một ngày. Nếu gặp vấn đề, hãy mở issue để nhóm có thể xem xét sớm hơn.

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Khi cài từ mã nguồn, bạn cũng có thể chỉ định một nhánh cụ thể. Cách này hữu ích khi muốn thử tính năng hoặc bản sửa lỗi chưa được phát hành:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

Sau khi cài đặt, hãy [kiểm tra](#check-installation) để xác nhận mọi thứ hoạt động đúng.

### Cài đặt ở chế độ editable

Cài đặt từ mã nguồn cũng cho phép bạn thiết lập [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/). Đây là cách cài đặt nâng cao, phù hợp nếu bạn định đóng góp cho huggingface_hub và cần kiểm thử thay đổi. Trước tiên, hãy clone một bản repo về máy:

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

Các lệnh này liên kết thư mục repo bạn đã clone với đường dẫn thư viện Python. Từ đó, Python sẽ tìm gói trong thư mục này bên cạnh các đường dẫn thư viện thông thường. Ví dụ, nếu các gói Python thường nằm trong ./.venv/lib/python3.13/site-packages/, Python cũng sẽ tìm trong ./huggingface_hub/.

## Cài đặt Hugging Face CLI

Dùng trình cài đặt một dòng lệnh để thiết lập CLI hf mà không làm thay đổi môi trường Python:

Trên macOS và Linux:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Trên Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Để nâng cấp bản đã cài, chạy hf update — lệnh sẽ nhận diện cách cài (trình cài đặt độc lập, Homebrew hoặc pip) và dùng đúng phương thức tương ứng.

## Cài đặt bằng conda

Nếu quen dùng conda, bạn có thể cài huggingface_hub từ [kênh conda-forge](https://anaconda.org/conda-forge/huggingface_hub):

```bash
conda install -c conda-forge huggingface_hub
```

Sau khi cài đặt, hãy [kiểm tra](#check-installation) để xác nhận mọi thứ hoạt động đúng.

<a id="check-installation"></a>

## Kiểm tra cài đặt

Sau khi cài đặt, chạy lệnh sau để kiểm tra huggingface_hub hoạt động bình thường:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

Lệnh này lấy thông tin trên Hub về mô hình [gpt2](https://huggingface.co/gpt2). Kết quả sẽ có dạng:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Giới hạn trên Windows

Mục tiêu của dự án là phổ cập các công cụ ML hữu ích trên mọi nền tảng. Vì vậy, huggingface_hub được xây dựng để hoạt động trên cả hệ điều hành Unix và Windows. Tuy nhiên, thư viện vẫn có một số giới hạn trên Windows. Dưới đây là danh sách các vấn đề đã biết. Nếu gặp vấn đề chưa được ghi nhận, vui lòng [mở issue trên GitHub](https://github.com/huggingface/huggingface_hub/issues/new/choose).

- Hệ thống cache của huggingface_hub dùng symbolic link để lưu tệp tải từ Hub hiệu quả hơn. Trên Windows, bạn cần bật Developer Mode hoặc chạy script với quyền quản trị viên để tạo symbolic link. Nếu chưa bật, cache vẫn hoạt động nhưng không được tối ưu. Xem thêm phần [giới hạn của cache](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#limitations).
- Đường dẫn tệp trên Hub có thể chứa ký tự đặc biệt (ví dụ: "path/to?/my/file"). Windows hạn chế một số [ký tự trong tên tệp](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names), nên không thể tải các tệp có đường dẫn như vậy xuống Windows. Trường hợp này hiếm gặp. Nếu cho rằng đây là lỗi, hãy liên hệ chủ repo hoặc nhóm dự án để cùng tìm hiểu.

## Bước tiếp theo

Sau khi cài đặt huggingface_hub, bạn có thể [cấu hình biến môi trường](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) hoặc xem một trong [các hướng dẫn thực hành (tiếng Anh)](https://huggingface.co/docs/huggingface_hub/en/guides/overview) để bắt đầu.
