import json
import zipfile
from pathlib import Path
from typing import Iterable, Tuple, Union

import pytest
from pytest_mock import MockerFixture

from huggingface_hub.errors import DDUFExportError, DDUFInvalidEntryNameError
from huggingface_hub.serialization._dduf import (
    DDUFEntry,
    _validate_dduf_entry_name,
    export_entries_as_dduf,
    export_folder_as_dduf,
    read_dduf_file,
)


class TestDDUFEntry:
    @pytest.fixture
    def dummy_entry(self, tmp_path: Path) -> DDUFEntry:
        dummy_dduf = tmp_path / "dummy_dduf.dduf"
        dummy_dduf.write_bytes(b"somethingCONTENTsomething")
        return DDUFEntry(filename="dummy.json", length=7, offset=9, dduf_path=dummy_dduf)

    def test_dataclass(self, dummy_entry: DDUFEntry):
        assert dummy_entry.filename == "dummy.json"
        assert dummy_entry.length == 7
        assert dummy_entry.offset == 9
        assert str(dummy_entry.dduf_path).endswith("dummy_dduf.dduf")

    def test_read_text(self, dummy_entry: DDUFEntry):
        assert dummy_entry.read_text() == "CONTENT"

    def test_as_mmap(self, dummy_entry: DDUFEntry):
        with dummy_entry.as_mmap() as mmap:
            assert mmap == b"CONTENT"


class TestUtils:
    @pytest.mark.parametrize("filename", ["dummy.txt", "dummy.json", "dummy.safetensors"])
    def test_entry_name_valid_extension(self, filename: str):
        assert _validate_dduf_entry_name(filename) == filename

    @pytest.mark.parametrize("filename", ["dummy", "dummy.bin", "dummy.dduf"])
    def test_entry_name_invalid_extension(self, filename: str):
        with pytest.raises(DDUFInvalidEntryNameError):
            _validate_dduf_entry_name(filename)

    @pytest.mark.parametrize("filename", ["encoder\\dummy.json", "C:\\dummy.json"])
    def test_entry_name_no_windows_path(self, filename: str):
        with pytest.raises(DDUFInvalidEntryNameError):
            _validate_dduf_entry_name(filename)

    def test_entry_name_stripped(
        self,
    ):
        assert _validate_dduf_entry_name("/dummy.json") == "dummy.json"

    def test_entry_name_no_nested_directory(self):
        _validate_dduf_entry_name("bar/dummy.json")  # 1 level is ok
        with pytest.raises(DDUFInvalidEntryNameError):
            _validate_dduf_entry_name("foo/bar/dummy.json")  # not more


class TestExportFolder:
    @pytest.fixture
    def dummy_folder(self, tmp_path: Path):
        folder_path = tmp_path / "dummy_folder"
        folder_path.mkdir()
        encoder_path = folder_path / "encoder"
        encoder_path.mkdir()
        subdir_path = encoder_path / "subdir"
        subdir_path.mkdir()

        (folder_path / "config.json").touch()
        (folder_path / "model.safetensors").touch()
        (folder_path / "model.bin").touch()  # won't be included
        (encoder_path / "config.json").touch()
        (encoder_path / "model.safetensors").touch()
        (encoder_path / "model.bin").touch()  # won't be included
        (subdir_path / "config.json").touch()  # won't be included
        return folder_path

    def test_export_folder(self, dummy_folder: Path, mocker: MockerFixture):
        mock = mocker.patch("huggingface_hub.serialization._dduf.export_entries_as_dduf")
        export_folder_as_dduf("dummy.dduf", dummy_folder)
        mock.assert_called_once()
        args = mock.call_args_list[0].args

        assert args[0] == "dummy.dduf"
        assert list(args[1]) == [
            # args[1] is a generator of tuples (path_in_archive, path_on_disk)
            ("config.json", dummy_folder / "config.json"),
            ("model.safetensors", dummy_folder / "model.safetensors"),
            ("encoder/config.json", dummy_folder / "encoder/config.json"),
            ("encoder/model.safetensors", dummy_folder / "encoder/model.safetensors"),
        ]


class TestExportEntries:
    @pytest.fixture
    def dummy_entries(self, tmp_path: Path) -> Iterable[Tuple[str, Union[str, Path, bytes]]]:
        (tmp_path / "config.json").write_text(json.dumps({"foo": "bar"}))
        (tmp_path / "does_have_to_be_same_name.safetensors").write_bytes(b"this is safetensors content")

        return [
            ("config.json", str(tmp_path / "config.json")),  # string path
            ("model.safetensors", tmp_path / "does_have_to_be_same_name.safetensors"),  # pathlib path
            ("hello.txt", b"hello world"),  # raw bytes
        ]

    def test_export_entries(self, tmp_path: Path, dummy_entries: Iterable[Tuple[str, Union[str, Path, bytes]]]):
        export_entries_as_dduf(tmp_path / "dummy.dduf", dummy_entries)

        with zipfile.ZipFile(tmp_path / "dummy.dduf", "r") as archive:
            assert archive.compression == zipfile.ZIP_STORED  # uncompressed!
            assert archive.namelist() == ["config.json", "model.safetensors", "hello.txt"]
            assert archive.read("config.json") == b'{"foo": "bar"}'
            assert archive.read("model.safetensors") == b"this is safetensors content"
            assert archive.read("hello.txt") == b"hello world"

    def test_export_entries_invalid_name(self, tmp_path: Path):
        with pytest.raises(DDUFExportError, match="Invalid entry name") as e:
            export_entries_as_dduf(tmp_path / "dummy.dduf", [("config", "config.json")])
        assert isinstance(e.value.__cause__, DDUFInvalidEntryNameError)

    def test_export_entries_no_duplicate(self, tmp_path: Path):
        with pytest.raises(DDUFExportError, match="Can't add duplicate entry"):
            export_entries_as_dduf(
                tmp_path / "dummy.dduf", [("config.json", b"content1"), ("config.json", b"content2")]
            )


class TestReadDDUFFile:
    @pytest.fixture
    def dummy_dduf_file(self, tmp_path: Path) -> Path:
        with zipfile.ZipFile(tmp_path / "dummy.dduf", "w") as archive:
            archive.writestr("config.json", b'{"foo": "bar"}')
            archive.writestr("model.safetensors", b"this is safetensors content")
            archive.writestr("hello.txt", b"hello world")
        return tmp_path / "dummy.dduf"

    def test_read_dduf_file(self, dummy_dduf_file: Path):
        entries = read_dduf_file(dummy_dduf_file)
        assert len(entries) == 3
        config_entry = entries["config.json"]
        model_entry = entries["model.safetensors"]
        hello_entry = entries["hello.txt"]

        assert config_entry.filename == "config.json"
        assert config_entry.dduf_path == dummy_dduf_file
        assert config_entry.read_text() == '{"foo": "bar"}'
        with dummy_dduf_file.open("rb") as f:
            f.seek(config_entry.offset)
            assert f.read(config_entry.length) == b'{"foo": "bar"}'

        assert model_entry.filename == "model.safetensors"
        assert model_entry.dduf_path == dummy_dduf_file
        assert model_entry.read_text() == "this is safetensors content"
        with dummy_dduf_file.open("rb") as f:
            f.seek(model_entry.offset)
            assert f.read(model_entry.length) == b"this is safetensors content"

        assert hello_entry.filename == "hello.txt"
        assert hello_entry.dduf_path == dummy_dduf_file
        assert hello_entry.read_text() == "hello world"
        with dummy_dduf_file.open("rb") as f:
            f.seek(hello_entry.offset)
            assert f.read(hello_entry.length) == b"hello world"
