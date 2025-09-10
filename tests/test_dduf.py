import json
import zipfile
from pathlib import Path
from typing import Iterable, Union

import pytest
from pytest_mock import MockerFixture

from huggingface_hub.errors import DDUFCorruptedFileError, DDUFExportError, DDUFInvalidEntryNameError
from huggingface_hub.serialization._dduf import (
    DDUFEntry,
    _load_content,
    _validate_dduf_entry_name,
    _validate_dduf_structure,
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

    @pytest.mark.parametrize("filename", ["dummy", "dummy.bin", "dummy.dduf", "dummy.gguf"])
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

    def test_load_content(self, tmp_path: Path):
        content = b"hello world"
        path = tmp_path / "hello.txt"
        path.write_bytes(content)

        assert _load_content(content) == content  # from bytes
        assert _load_content(path) == content  # from Path
        assert _load_content(str(path)) == content  # from str

    def test_validate_dduf_structure_valid(self):
        _validate_dduf_structure(
            {  # model_index.json content
                "_some_key": "some_value",
                "encoder": {
                    "config.json": {},
                    "model.safetensors": {},
                },
            },
            {  # entries in DDUF archive
                "model_index.json",
                "something.txt",
                "encoder/config.json",
                "encoder/model.safetensors",
            },
        )

    def test_validate_dduf_structure_not_a_dict(self):
        with pytest.raises(DDUFCorruptedFileError, match="Must be a dictionary."):
            _validate_dduf_structure(["not a dict"], {})  # content from 'model_index.json'

    def test_validate_dduf_structure_missing_folder(self):
        with pytest.raises(DDUFCorruptedFileError, match="Missing required entry 'encoder' in 'model_index.json'."):
            _validate_dduf_structure({}, {"encoder/config.json", "encoder/model.safetensors"})

    def test_validate_dduf_structure_missing_config_file(self):
        with pytest.raises(DDUFCorruptedFileError, match="Missing required file in folder 'encoder'."):
            _validate_dduf_structure(
                {"encoder": {}},
                {
                    "encoder/not_a_config.json",  # expecting a config.json / tokenizer_config.json / preprocessor_config.json / scheduler_config.json
                    "encoder/model.safetensors",
                },
            )


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
        assert sorted(list(args[1])) == [
            # args[1] is a generator of tuples (path_in_archive, path_on_disk)
            ("config.json", dummy_folder / "config.json"),
            ("encoder/config.json", dummy_folder / "encoder/config.json"),
            ("encoder/model.safetensors", dummy_folder / "encoder/model.safetensors"),
            ("model.safetensors", dummy_folder / "model.safetensors"),
        ]


class TestExportEntries:
    @pytest.fixture
    def dummy_entries(self, tmp_path: Path) -> Iterable[tuple[str, Union[str, Path, bytes]]]:
        (tmp_path / "model_index.json").write_text(json.dumps({"foo": "bar"}))
        (tmp_path / "doesnt_have_to_be_same_name.safetensors").write_bytes(b"this is safetensors content")

        return [
            ("model_index.json", str(tmp_path / "model_index.json")),  # string path
            ("model.safetensors", tmp_path / "doesnt_have_to_be_same_name.safetensors"),  # pathlib path
            ("hello.txt", b"hello world"),  # raw bytes
        ]

    def test_export_entries(
        self, tmp_path: Path, dummy_entries: Iterable[tuple[str, Union[str, Path, bytes]]], mocker: MockerFixture
    ):
        mock = mocker.patch("huggingface_hub.serialization._dduf._validate_dduf_structure")
        export_entries_as_dduf(tmp_path / "dummy.dduf", dummy_entries)
        mock.assert_called_once_with({"foo": "bar"}, {"model_index.json", "model.safetensors", "hello.txt"})

        with zipfile.ZipFile(tmp_path / "dummy.dduf", "r") as archive:
            assert archive.compression == zipfile.ZIP_STORED  # uncompressed!
            assert archive.namelist() == ["model_index.json", "model.safetensors", "hello.txt"]
            assert archive.read("model_index.json") == b'{"foo": "bar"}'
            assert archive.read("model.safetensors") == b"this is safetensors content"
            assert archive.read("hello.txt") == b"hello world"

    def test_export_entries_invalid_name(self, tmp_path: Path):
        with pytest.raises(DDUFExportError, match="Invalid entry name") as e:
            export_entries_as_dduf(tmp_path / "dummy.dduf", [("config", "model_index.json")])
        assert isinstance(e.value.__cause__, DDUFInvalidEntryNameError)

    def test_export_entries_no_duplicate(self, tmp_path: Path):
        with pytest.raises(DDUFExportError, match="Can't add duplicate entry"):
            export_entries_as_dduf(
                tmp_path / "dummy.dduf",
                [
                    ("model_index.json", b'{"key": "content1"}'),
                    ("model_index.json", b'{"key": "content2"}'),
                ],
            )

    def test_export_entries_model_index_required(self, tmp_path: Path):
        with pytest.raises(DDUFExportError, match="Missing required 'model_index.json' entry"):
            export_entries_as_dduf(tmp_path / "dummy.dduf", [("model.safetensors", b"content")])


class TestReadDDUFFile:
    @pytest.fixture
    def dummy_dduf_file(self, tmp_path: Path) -> Path:
        with zipfile.ZipFile(tmp_path / "dummy.dduf", "w") as archive:
            archive.writestr("model_index.json", b'{"foo": "bar"}')
            archive.writestr("model.safetensors", b"this is safetensors content")
            archive.writestr("hello.txt", b"hello world")
        return tmp_path / "dummy.dduf"

    def test_read_dduf_file(self, dummy_dduf_file: Path, mocker: MockerFixture):
        mock = mocker.patch("huggingface_hub.serialization._dduf._validate_dduf_structure")

        entries = read_dduf_file(dummy_dduf_file)
        assert len(entries) == 3
        index_entry = entries["model_index.json"]
        model_entry = entries["model.safetensors"]
        hello_entry = entries["hello.txt"]

        mock.assert_called_once_with({"foo": "bar"}, {"model_index.json", "model.safetensors", "hello.txt"})

        assert index_entry.filename == "model_index.json"
        assert index_entry.dduf_path == dummy_dduf_file
        assert index_entry.read_text() == '{"foo": "bar"}'
        with dummy_dduf_file.open("rb") as f:
            f.seek(index_entry.offset)
            assert f.read(index_entry.length) == b'{"foo": "bar"}'

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

    def test_model_index_required(self, tmp_path: Path):
        with zipfile.ZipFile(tmp_path / "dummy.dduf", "w") as archive:
            archive.writestr("model.safetensors", b"this is safetensors content")
        with pytest.raises(DDUFCorruptedFileError, match="Missing required 'model_index.json' entry"):
            read_dduf_file(tmp_path / "dummy.dduf")
