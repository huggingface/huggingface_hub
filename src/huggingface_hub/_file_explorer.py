import mmap
import os
from contextlib import contextmanager
from typing import Dict, Generator, Protocol, Union, runtime_checkable

from .serialization._dduf import DDUFEntry, read_dduf_file


def get_file_explorer(item: Union[str, os.PathLike, DDUFEntry]) -> "FileExplorer":
    if isinstance(item, FileExplorer):
        return item
    if isinstance(item, DDUFEntry):
        return DDUFFileExplorer({"": item})
    if str(item).endswith(".dduf"):
        return DDUFFileExplorer(read_dduf_file(item))
    return DiskFileExplorer(item)


@runtime_checkable
class FileExplorer(Protocol):
    def navigate_to(self, *paths: str) -> "FileExplorer": ...
    def listdir(self, *paths: str) -> list[str]: ...
    def is_dir(self, *paths: str) -> bool: ...
    def is_file(self, *paths: str) -> bool: ...
    def file_extension(self) -> str: ...
    def exists(self, *paths: str) -> bool: ...
    def read_text(self, *paths: str, encoding: str = "utf-8") -> str: ...
    @contextmanager
    def as_mmap(self, *paths: str) -> Generator[bytes, None, None]: ...


class DiskFileExplorer:
    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self.root = str(root)

    def navigate_to(self, *paths: str) -> "FileExplorer":
        return DiskFileExplorer(self._path(*paths))

    def listdir(self, *paths: str) -> list[str]:
        return os.listdir(self._path(*paths))

    def is_dir(self, *paths: str) -> bool:
        return os.path.isdir(self._path(*paths))

    def is_file(self, *paths: str) -> bool:
        return os.path.isfile(self._path(*paths))

    def file_extension(self) -> str:
        if not self.is_file():
            raise ValueError("Cannot get file extension: not a file")
        return os.path.splitext(self.root)[1]

    def exists(self, *paths: str) -> bool:
        return os.path.exists(self._path(*paths))

    def read_text(self, *paths: str, encoding: str = "utf-8") -> str:
        with open(self._path(*paths), "r", encoding=encoding) as f:
            return f.read()

    @contextmanager
    def as_mmap(self, *paths: str) -> Generator[bytes, None, None]:
        with open(self._path(*paths), "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                yield m.read()

    def _path(self, *paths: str) -> str:
        return os.path.join(self.root, *paths)

    def __repr__(self):
        return f"<DiskFileExplorer(root={self.root!r})>"


class DDUFFileExplorer:
    def __init__(self, dduf_entries: Dict[str, DDUFEntry]) -> None:
        # dduf_entries is a dictionary of file paths to DDUFEntry objects
        # paths are relative to the root of the DDUF
        # paths are always normalized to use "/" as the separator and not start with "/"
        self.dduf_entries = dduf_entries

    def navigate_to(self, *paths: str) -> "FileExplorer":
        if len(paths) == 0:
            return self
        queried_name = self._entry_name(*paths)
        return DDUFFileExplorer(
            {
                entry_name[len(queried_name) :].strip("/"): entry
                for entry_name, entry in self.dduf_entries.items()
                if entry_name.startswith(queried_name + "/") or entry_name == queried_name
            }
        )

    def listdir(self, *paths: str) -> list[str]:
        if len(paths) == 0:
            # return only the top level files
            return [filename for filename in self.dduf_entries if not filename.strip("/").count("/")]
        return [filename for filename in self.dduf_entries if filename.startswith(self._entry_name(*paths) + "/")]

    def is_dir(self, *paths: str) -> bool:
        if list(self.dduf_entries.keys()) == [""]:
            # means navigated to a file
            return False
        if len(paths) == 0:
            return len(self.dduf_entries) > 0
        return any(filename.startswith(self._entry_name(*paths) + "/") for filename in self.dduf_entries)

    def is_file(self, *paths: str) -> bool:
        return self._entry_name(*paths) in self.dduf_entries

    def file_extension(self) -> str:
        if len(self.dduf_entries) != 1:
            raise ValueError("Cannot get file extension: not a file.")
        return os.path.splitext(list(self.dduf_entries.keys())[0])[1]

    def exists(self, *paths: str) -> bool:
        return self.is_dir(*paths) or self.is_file(*paths)

    def read_text(self, *paths: str, encoding: str = "utf-8") -> str:
        return self.dduf_entries[self._entry_name(*paths)].read_text(encoding)

    @contextmanager
    def as_mmap(self, *paths: str) -> Generator[bytes, None, None]:
        with self.dduf_entries[self._entry_name(*paths)].as_mmap() as m:
            yield m

    def _entry_name(self, *paths: str) -> str:
        return "/".join(path for path in paths if path).strip("/")

    def __repr__(self):
        if len(self.dduf_entries) == 0:
            return "<DDUFFileExplorer()>"
        first = next(iter(self.dduf_entries.values()))
        root = first.dduf_path
        if len(self.dduf_entries) == 1:
            return f"<DDUFFileExplorer root={root} entry={first.filename}>"
        return f"<DDUFFileExplorer root={root}>"
