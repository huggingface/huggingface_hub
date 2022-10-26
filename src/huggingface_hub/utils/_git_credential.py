import subprocess
from contextlib import contextmanager
from typing import IO, Generator, Optional, Tuple, Union

from ..constants import ENDPOINT


def write_to_credential_store(username: str, password: str) -> None:
    with _interactive_subprocess("git credential-store store") as (stdin, _):
        input_username = f"username={username.lower()}"
        input_password = f"password={password}"

        stdin.write(
            f"url={ENDPOINT}\n{input_username}\n{input_password}\n\n".encode("utf-8")
        )
        stdin.flush()


def read_from_credential_store(
    username: Optional[str] = None,
) -> Union[Tuple[str, str], Tuple[None, None]]:
    """
    Reads the credential store relative to huggingface.co.

    Args:
        username (`str`, *optional*):
            A username to filter to search. If not specified, the first entry under
            `huggingface.co` endpoint is returned.

    Returns:
        `Tuple[str, str]` or `Tuple[None, None]`: either a username/password pair or
        None/None if credential has not been found. The returned username is always
        lowercase.
    """
    with _interactive_subprocess("git credential-store get") as (stdin, stdout):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input.encode("utf-8"))
        stdin.flush()
        output = stdout.read().decode("utf-8")

    if len(output) == 0:
        return None, None

    username, password = [line for line in output.split("\n") if len(line) != 0]
    return username.split("=")[1], password.split("=")[1]


def erase_from_credential_store(username: Optional[str] = None) -> None:
    """
    Erases the credential store relative to huggingface.co.

    Args:
        username (`str`, *optional*):
            A username to filter to search. If not specified, all entries under
            `huggingface.co` endpoint is erased.
    """
    with _interactive_subprocess("git credential-store erase") as (stdin, _):
        standard_input = f"url={ENDPOINT}\n"
        if username is not None:
            standard_input += f"username={username.lower()}\n"
        standard_input += "\n"

        stdin.write(standard_input.encode("utf-8"))
        stdin.flush()


@contextmanager
def _interactive_subprocess(
    command: str,
) -> Generator[Tuple[IO[bytes], IO[bytes]], None, None]:
    with subprocess.Popen(
        command.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        assert process.stdin is not None, "subprocess is opened as subprocess.PIPE"
        assert process.stdout is not None, "subprocess is opened as subprocess.PIPE"
        yield process.stdin, process.stdout
