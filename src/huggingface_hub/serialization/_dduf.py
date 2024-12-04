import logging
import mmap
import shutil
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Union

from ..errors import DDUFCorruptedFileError


logger = logging.getLogger(__name__)

DDUF_ALLOWED_ENTRIES = {".json", ".gguf", ".txt", ".safetensors"}


@dataclass
class DDUFEntry:
    """Object representing a file entry in a DDUF file.

    See [`read_dduf_file`] for how to read a DDUF file.

    Attributes:
        filename (str):
            The name of the file in the DDUF archive.
        offset (int):
            The offset of the file in the DDUF archive.
        length (int):
            The length of the file in the DDUF archive.
        dduf_path (str):
            The path to the DDUF archive (for internal use).
    """

    filename: str
    length: int
    offset: int

    dduf_path: Path = field(repr=False)

    @contextmanager
    def as_mmap(self) -> Generator[bytes, None, None]:
        """Open the file as a memory-mapped file.

        Useful to load safetensors directly from the file.

        Example:
            ```py
            >>> import safetensors.torch
            >>> with entry.as_mmap() as mm:
            ...     tensors = safetensors.torch.load(mm)
            ```
        """
        with self.dduf_path.open("rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                yield mm[self.offset : self.offset + self.length]

    def read_text(self, encoding="utf-8") -> str:
        """Read the file as text.

        Useful for '.txt' and '.json' entries.
        """
        with self.dduf_path.open("rb") as f:
            f.seek(self.offset)
            return f.read(self.length).decode(encoding=encoding)


def read_dduf_file(dduf_path: Union[Path, str]) -> Dict[str, DDUFEntry]:
    """
    Read a DDUF file and return a dictionary of entries.

    Only the metadata is read, the data is not loaded in memory.

    Args:
        dduf_path (`str` or `Path`):
            The path to the DDUF file to read.

    Returns:
        `Dict[str, DDUFEntry]`:
            A dictionary of [`DDUFEntry`] indexed by filename.

    Raises:
        - [`DDUFCorruptedFileError`]: If the DDUF file is corrupted (i.e. doesn't follow the DDUF format).
    """
    entries = {}
    dduf_path = Path(dduf_path)
    with zipfile.ZipFile(str(dduf_path), "r") as zf:
        for info in zf.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise DDUFCorruptedFileError("Data must not be compressed in GGUF file.")

            # Use private attribute to get data range for this file.
            # Let's reconsider later if it's too problematic (worse case, we can build our own metadata parser).
            # Note: simply doing `info.header_offset + len(info.FileHeader())` doesn't work because of the ZIP64 extension.
            offset = info._end_offset - info.compress_size

            entries[info.filename] = DDUFEntry(
                filename=info.filename, offset=offset, length=info.file_size, dduf_path=dduf_path
            )
    return entries


def write_dduf_file(dduf_path: Union[str, Path], diffuser_path: Union[str, Path]) -> None:
    """
    Write a DDUF file from a diffusers folder.

    A DDUF file is simply a ZIP archive with a few constraints (force ZIP64, no compression, only certain files).

    Args:
        dduf_path (`str` or `Path`):
            The path to the DDUF file to write.
        diffuser_path (`str` or `Path`):
            The path to the folder containing the diffusers model.
    """
    # TODO: update method signature.
    #       DDUF filename should be inferred as much as possible from high-level info (precision, model, etc.) to ensure consistency.
    #       Example: "stable-diffusion-3.5-Q4-BNB.dduf"
    #       See https://github.com/huggingface/diffusers/pull/10037#discussion_r1862275730.
    logger.info("Writing DDUF file %s from folder %s", dduf_path, diffuser_path)
    diffuser_path = Path(diffuser_path)
    with zipfile.ZipFile(str(dduf_path), "w", zipfile.ZIP_STORED) as archive:
        for path in diffuser_path.glob("**/*"):
            if path.is_dir():
                logger.debug("Skipping directory %s", path)
                continue
            if path.suffix not in DDUF_ALLOWED_ENTRIES:
                logger.debug("Skipping file %s", path)
                continue
            logger.debug("Adding file %s", path)
            with archive.open(str(path.relative_to(diffuser_path)), "w", force_zip64=True) as f:
                with path.open("rb") as src:
                    shutil.copyfileobj(src, f, 1024 * 8)
    logger.info("Done writing DDUF file %s", dduf_path)
