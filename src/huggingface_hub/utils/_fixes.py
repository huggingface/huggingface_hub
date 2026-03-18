import contextlib
import os
import shutil
import stat
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Optional, Union

import yaml
from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout

from .. import constants
from . import logging


logger = logging.get_logger(__name__)

# Wrap `yaml.dump` to set `allow_unicode=True` by default.
#
# Example:
# ```py
# >>> yaml.dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "\\U0001F440"\nsome unicode: "\\u65E5\\u672C\\u304B"\n'
#
# >>> yaml_dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "👀"\nsome unicode: "日本か"\n'
# ```
yaml_dump: Callable[..., str] = partial(yaml.dump, stream=None, allow_unicode=True)  # type: ignore


@contextlib.contextmanager
def SoftTemporaryDirectory(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[Union[Path, str]] = None,
    **kwargs,
) -> Generator[Path, None, None]:
    """
    Context manager to create a temporary directory and safely delete it.

    If tmp directory cannot be deleted normally, we set the WRITE permission and retry.
    If cleanup still fails, we give up but don't raise an exception. This is equivalent
    to  `tempfile.TemporaryDirectory(..., ignore_cleanup_errors=True)` introduced in
    Python 3.10.

    See https://www.scivision.dev/python-tempfile-permission-error-windows/.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield Path(tmpdir.name).resolve()

    try:
        # First once with normal cleanup
        shutil.rmtree(tmpdir.name)
    except Exception:
        # If failed, try to set write permission and retry
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        except Exception:
            pass

    # And finally, cleanup the tmpdir.
    # If it fails again, give up but do not throw error
    try:
        tmpdir.cleanup()
    except Exception:
        pass


def _set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


@contextlib.contextmanager
def WeakFileLock(
    lock_file: Union[str, Path], *, timeout: Optional[float] = None, lifetime: Optional[float] = None
) -> Generator[BaseFileLock, None, None]:
    """A filelock with some custom logic.

    This filelock is weaker than the default filelock in that:
    1. It won't raise an exception if release fails.
    2. It will default to a SoftFileLock if the filesystem does not support flock.
    3. Lock files are created with mode 0o664 (group-writable) instead of the default 0o644.
       This allows multiple users sharing a cache directory to wait for locks.

    An INFO log message is emitted every 10 seconds if the lock is not acquired immediately.
    If a timeout is provided, a `filelock.Timeout` exception is raised if the lock is not acquired within the timeout.

    If a lifetime is provided, it enables stale lock recovery: the lock holder periodically
    updates the lock file's mtime (heartbeat), and other processes consider the lock stale
    if the mtime is older than the lifetime. This handles the case where a process crashes
    (e.g. OOM kill in Kubernetes) while holding a lock on shared filesystems.

    The timeout can also be configured globally via the `HF_HUB_LOCK_TIMEOUT` environment variable.
    """
    # Allow env var override for timeout (explicit parameter takes precedence)
    if timeout is None and constants.HF_HUB_LOCK_TIMEOUT is not None:
        timeout = float(constants.HF_HUB_LOCK_TIMEOUT)

    log_interval = constants.FILELOCK_LOG_EVERY_SECONDS

    # lifetime parameter requires filelock>=3.24.0; gracefully degrade if unavailable
    lock_kwargs: dict = {"timeout": log_interval, "mode": 0o664}
    if lifetime is not None:
        lock_kwargs["lifetime"] = lifetime
    try:
        lock = FileLock(lock_file, **lock_kwargs)
    except TypeError:
        # filelock version doesn't support lifetime - fall back without it
        lock_kwargs.pop("lifetime", None)
        lock = FileLock(lock_file, **lock_kwargs)
        if lifetime is not None:
            logger.debug("filelock version does not support 'lifetime' parameter. Stale lock detection disabled.")
            lifetime = None  # disable heartbeat too since lifetime won't be enforced

    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            logger.warning(
                f"Lock acquisition timed out after {timeout:.0f}s on {lock_file}. "
                f"This may be caused by a stale lock from a crashed process. "
                f"If you are sure no other process is downloading, you can manually delete the lock file: {lock_file}"
            )
            raise Timeout(str(lock_file))

        try:
            lock.acquire(timeout=min(log_interval, timeout - elapsed_time) if timeout else log_interval)
        except Timeout:
            logger.info(
                f"Still waiting to acquire lock on {lock_file} (elapsed: {time.time() - start_time:.1f} seconds)"
            )
        except NotImplementedError as e:
            if "use SoftFileLock instead" in str(e):
                logger.warning(
                    "FileSystem does not appear to support flock. Falling back to SoftFileLock for %s", lock_file
                )
                soft_kwargs: dict = {"timeout": log_interval}
                if lifetime is not None:
                    soft_kwargs["lifetime"] = lifetime
                try:
                    lock = SoftFileLock(lock_file, **soft_kwargs)
                except TypeError:
                    lock = SoftFileLock(lock_file, timeout=log_interval)
                    lifetime = None  # disable heartbeat if lifetime not supported
                continue
        else:
            break

    # Start heartbeat thread to periodically touch the lock file.
    # This keeps the lock's mtime fresh so other processes don't consider it stale.
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    if lifetime is not None:
        heartbeat_interval = min(lifetime / 3, 60)

        def _heartbeat():
            while not heartbeat_stop.wait(heartbeat_interval):
                try:
                    os.utime(str(lock_file))
                except OSError:
                    break

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

    try:
        yield lock
    finally:
        if heartbeat_thread is not None:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)
        try:
            lock.release()
        except OSError:
            try:
                Path(lock_file).unlink()
            except OSError:
                pass
