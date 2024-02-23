from pathlib import Path


def check_and_update_file_content(file: Path, expected_content: str, update: bool):
    content = file.read_text() if file.exists() else None
    if content != expected_content:
        if update:
            file.write_text(expected_content)
            print(f"  {file} has been updated. Please make sure the changes are accurate and commit them.")
        else:
            print(f"❌ Expected content mismatch in {file}.")
            exit(1)
