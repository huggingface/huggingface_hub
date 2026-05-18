import contextlib
import ctypes
import ctypes.util
import functools
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path

import yaml
from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout

from .. import constants
from . import logging


logger = logging.get_logger(__name__)


# Linux statfs(2) f_type magic numbers (see <linux/magic.h>). Filesystems where `flock(2)` is known
# to silently succeed for every caller without serializing — making `WeakFileLock` a no-op and
# corrupting concurrent cache writes. See `HF-MODEL-READ-FIX.md` and:
# - https://github.com/tox-dev/filelock/issues/389  (FileLock not work on gpfs)
# - https://github.com/huggingface/transformers/issues/30859
_LUSTRE_SUPER_MAGIC = 0x0BD00BD0
_GPFS_SUPER_MAGIC = 0x47504653
_NFS_SUPER_MAGIC = 0x6969
_SMB_SUPER_MAGIC = 0x517B
_CIFS_SUPER_MAGIC = 0xFF534D42
# FUSE is intentionally NOT in the broken set — its backend can be anything (Lustre, S3, sshfs, ...)
# so we let the runtime probe decide.
_KNOWN_BROKEN_FOR_FLOCK = frozenset(
    {
        _LUSTRE_SUPER_MAGIC,
        _GPFS_SUPER_MAGIC,
        _NFS_SUPER_MAGIC,
        _SMB_SUPER_MAGIC,
        _CIFS_SUPER_MAGIC,
    }
)

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
    suffix: str | None = None,
    prefix: str | None = None,
    dir: Path | str | None = None,
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


def _fs_magic(path: str) -> int:
    """Return the `statfs(2)` `f_type` magic for the filesystem at `path`, or `0` on any error.

    Posix-only. Used as the primary, cross-node-correct signal for filesystems where `flock(2)`
    is silently broken (returns success for every caller). Returning `0` on non-posix or on any
    error means callers fall through to the runtime probe / default code path.
    """
    if os.name != "posix":
        return 0

    class _Statfs(ctypes.Structure):
        # Layout differs by libc/arch; we only need `f_type`, which is the first field on Linux.
        # Reserve 128 bytes total — comfortably larger than any known `struct statfs`.
        _fields_ = [("f_type", ctypes.c_long), ("_pad", ctypes.c_ubyte * 120)]

    try:
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_name, use_errno=True)
        s = _Statfs()
        if libc.statfs(os.fsencode(path), ctypes.byref(s)) != 0:
            return 0
        return int(s.f_type) & 0xFFFFFFFF
    except (OSError, AttributeError):
        return 0


def _flock_probe_child(probe_path: str, q) -> None:
    """Child-process worker for `_flock_actually_serializes`.

    Defined at module top-level so the `spawn` start method can pickle it (lambdas/closures
    can't be pickled, which would break on platforms that default to spawn).
    """
    import fcntl as _fcntl

    try:
        with open(probe_path, "r") as child_fd:
            try:
                _fcntl.flock(child_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                # Got the lock while the parent still holds it — `flock` is a no-op here.
                q.put("broken")
            except BlockingIOError:
                q.put("ok")
    except Exception:
        q.put("error")


@functools.lru_cache(maxsize=None)
def _flock_actually_serializes(cache_dir: str) -> bool:
    """Best-effort probe: does `fcntl.flock` actually serialize on `cache_dir`?

    Spawns a child process that re-opens a probe file in `cache_dir` and tries `LOCK_EX | LOCK_NB`
    while the parent still holds the lock. On a healthy local filesystem the child should be
    blocked (`BlockingIOError`); on a broken parallel filesystem it gets the lock too.

    This is a within-node sanity check, not a cross-node test, and is only meaningful as a
    secondary signal on top of `_fs_magic`-based detection. Fail-closed: any timeout, exception,
    or ambiguous result returns `False` (treat as broken → caller will pick `SoftFileLock`).
    """
    import fcntl
    import multiprocessing

    probe = os.path.join(cache_dir, ".hf-flock-probe")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        parent_fd = open(probe, "w")
    except OSError:
        return False  # Can't even create the probe file; fail closed.

    try:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            # Something else holds the lock on this exact path; can't make a determination.
            return False

        # Use an explicit `spawn` context so we don't inherit the parent's lock-holding fd via
        # `fork`. The child re-opens the path; on a sane FS its fresh open file description must
        # block on the parent's lock. Spawn is slower (~100-200ms) but pickle-safe and portable.
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_flock_probe_child, args=(probe, q))
        p.start()
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
            return False  # Timeout → fail closed.
        try:
            result = q.get_nowait()
        except Exception:
            result = "error"
        return result == "ok"
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        parent_fd.close()
        try:
            os.unlink(probe)
        except OSError:
            pass


@functools.lru_cache(maxsize=None)
def _should_use_soft_lock(cache_dir: str) -> bool:
    """Decide whether `WeakFileLock` should pick `SoftFileLock` over native `FileLock`.

    Order of precedence:
      1. `HF_HUB_USE_SOFT_FILELOCK=1` → always soft.
      2. `HF_HUB_FORCE_FLOCK=1` → always native (override for sites with cluster-wide flock).
      3. Non-posix → native (Windows uses `WindowsFileLock` internally; not affected).
      4. `_fs_magic(cache_dir)` is in `_KNOWN_BROKEN_FOR_FLOCK` → soft.
      5. Otherwise: run the runtime probe and fall back to soft if it fails.
    """
    if constants.HF_HUB_USE_SOFT_FILELOCK:
        return True
    if constants.HF_HUB_FORCE_FLOCK:
        return False
    if os.name != "posix":
        return False
    if _fs_magic(cache_dir) in _KNOWN_BROKEN_FOR_FLOCK:
        return True
    return not _flock_actually_serializes(cache_dir)


@contextlib.contextmanager
def WeakFileLock(lock_file: str | Path, *, timeout: float | None = None) -> Generator[BaseFileLock, None, None]:
    """A filelock with some custom logic.

    This filelock is weaker than the default filelock in that:
    1. It won't raise an exception if release fails.
    2. It will default to a SoftFileLock if the filesystem does not support flock.
       Detection happens up-front via `_should_use_soft_lock` (statfs-based for known broken
       families like Lustre/GPFS/NFS, plus a fail-closed runtime probe for unknown filesystems).
       The legacy `NotImplementedError` fallback inside the acquisition loop is kept as a
       belt-and-suspenders safety net.
    3. Lock files are created with mode 0o664 (group-writable) instead of the default 0o644.
       This allows multiple users sharing a cache directory to wait for locks.

    An INFO log message is emitted every 10 seconds if the lock is not acquired immediately.
    If a timeout is provided, a `filelock.Timeout` exception is raised if the lock is not acquired within the timeout.
    """
    log_interval = constants.FILELOCK_LOG_EVERY_SECONDS
    cache_dir = os.path.dirname(os.path.abspath(str(lock_file)))
    lock: BaseFileLock
    if _should_use_soft_lock(cache_dir):
        lock = SoftFileLock(lock_file, timeout=log_interval)
    else:
        lock = FileLock(lock_file, timeout=log_interval, mode=0o664)
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
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
                lock = SoftFileLock(lock_file, timeout=log_interval)
                continue
        else:
            break

    try:
        yield lock
    finally:
        try:
            lock.release()
        except OSError:
            try:
                Path(lock_file).unlink()
            except OSError:
                pass
