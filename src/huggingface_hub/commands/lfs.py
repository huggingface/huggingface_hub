"""
Implementation of a custom transfer agent for the transfer type "multipart" for
git-lfs.

Inspired by:
github.com/cbartz/git-lfs-swift-transfer-agent/blob/master/git_lfs_swift_transfer.py

Spec is: github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md


To launch debugger while developing:

``` [lfs "customtransfer.multipart"]
path = /path/to/huggingface_hub/.env/bin/python args = -m debugpy --listen 5678
--wait-for-client
/path/to/huggingface_hub/src/huggingface_hub/commands/huggingface_cli.py
lfs-multipart-upload ```"""

import concurrent.futures as futures
import json
import os
import subprocess
import sys
import threading
from argparse import _SubParsersAction
from typing import Dict, List, Optional

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.lfs import LFS_MULTIPART_UPLOAD_COMMAND, SliceFileObj

from ..utils import get_session, hf_raise_for_status, logging


logger = logging.get_logger(__name__)


class LfsCommands(BaseHuggingfaceCLICommand):
    """
    Implementation of a custom transfer agent for the transfer type "multipart"
    for git-lfs. This lets users upload large files >5GB 🔥. Spec for LFS custom
    transfer agent is:
    https://github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md

    This introduces two commands to the CLI:

    1. $ huggingface-cli lfs-enable-largefiles

    This should be executed once for each model repo that contains a model file
    >5GB. It's documented in the error message you get if you just try to git
    push a 5GB file without having enabled it before.

    2. $ huggingface-cli lfs-multipart-upload

    This command is called by lfs directly and is not meant to be called by the
    user.
    """

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        enable_parser = parser.add_parser(
            "lfs-enable-largefiles",
            help="Configure your repository to enable upload of files > 5GB.",
        )
        enable_parser.add_argument("path", type=str, help="Local path to repository you want to configure.")
        enable_parser.set_defaults(func=lambda args: LfsEnableCommand(args))

        upload_parser = parser.add_parser(
            LFS_MULTIPART_UPLOAD_COMMAND,
            help="Command will get called by git-lfs, do not call it directly.",
        )
        upload_parser.set_defaults(func=lambda args: LfsUploadCommand(args))


class LfsEnableCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        local_path = os.path.abspath(self.args.path)
        if not os.path.isdir(local_path):
            print("This does not look like a valid git repo.")
            exit(1)
        subprocess.run(
            "git config lfs.customtransfer.multipart.path huggingface-cli".split(),
            check=True,
            cwd=local_path,
        )
        subprocess.run(
            f"git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}".split(),
            check=True,
            cwd=local_path,
        )
        subprocess.run(
            "git config lfs.customtransfer.multipart.concurrent false".split(),
            check=True,
            cwd=local_path,
        )
        print("Local repo set up for largefiles")


def write_msg(msg: Dict):
    """Write out the message in Line delimited JSON."""
    msg_str = json.dumps(msg) + "\n"
    sys.stdout.write(msg_str)
    sys.stdout.flush()


def read_msg() -> Optional[Dict]:
    """Read Line delimited JSON from stdin."""
    msg = json.loads(sys.stdin.readline().strip())

    if "terminate" in (msg.get("type"), msg.get("event")):
        # terminate message received
        return None

    if msg.get("event") not in ("download", "upload"):
        logger.critical("Received unexpected message")
        sys.exit(1)

    return msg


class LfsUploadCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        # Immediately after invoking a custom transfer process, git-lfs
        # sends initiation data to the process over stdin.
        # This tells the process useful information about the configuration.
        init_msg = json.loads(sys.stdin.readline().strip())
        if not (init_msg.get("event") == "init" and init_msg.get("operation") == "upload"):
            write_msg({"error": {"code": 32, "message": "Wrong lfs init operation"}})
            sys.exit(1)

        ref_concurrency = max(init_msg.get("concurrenttransfers", 4), 1)

        # The transfer process should use the information it needs from the
        # initiation structure, and also perform any one-off setup tasks it
        # needs to do. It should then respond on stdout with a simple empty
        # confirmation structure, as follows:
        write_msg({})

        # After the initiation exchange, git-lfs will send any number of
        # transfer requests to the stdin of the transfer process, in a serial sequence.
        while True:
            msg = read_msg()
            if msg is None:
                # When all transfers have been processed, git-lfs will send
                # a terminate event to the stdin of the transfer process.
                # On receiving this message the transfer process should
                # clean up and terminate. No response is expected.
                sys.exit(0)

            oid = msg["oid"]
            filepath = msg["path"]
            completion_url = msg["action"]["href"]
            header = msg["action"]["header"]
            chunk_size = int(header.pop("chunk_size"))
            presigned_urls: List[str] = list(header.values())

            # Send a "started" progress event to allow other workers to start.
            # Otherwise they're delayed until first "progress" event is reported,
            # i.e. after the first 5GB by default (!)
            write_msg(
                {
                    "event": "progress",
                    "oid": oid,
                    "bytesSoFar": 1,
                    "bytesSinceLast": 0,
                }
            )

            # caculate workload per thread
            n_url = len(presigned_urls)
            n_thread = min(n_url, ref_concurrency)
            n_heavy = n_url % n_thread
            # n_light = n_thread - n_heavy
            n_per_light_thread = n_url // n_thread
            n_per_heavy_thread = n_per_light_thread + 1

            parts = []
            n_processed_bytes = 0
            lock = threading.Lock()
            err = False

            def _thread_process(start, n):
                nonlocal parts, n_processed_bytes, lock, err

                # open file for each thread
                with open(filepath, "rb") as file:
                    for i in range(start, start + n):
                        # cancel running if error occurred
                        if err:
                            return
                        presigned_url = presigned_urls[i]
                        with SliceFileObj(
                            file,
                            seek_from=i * chunk_size,
                            read_limit=chunk_size,
                        ) as data:
                            r = get_session().put(presigned_url, data=data)
                            hf_raise_for_status(r)

                            n_bytes = data.tell()
                            with lock:
                                parts.append(
                                    {
                                        "etag": r.headers.get("etag"),
                                        "partNumber": i + 1,
                                    }
                                )
                                n_processed_bytes += n_bytes

                                # the transfer process should post messages to stdout
                                write_msg(
                                    {
                                        "event": "progress",
                                        "oid": oid,
                                        "bytesSoFar": n_processed_bytes,
                                        "bytesSinceLast": n_bytes,
                                    }
                                )

            tasks = []
            executor = futures.ThreadPoolExecutor(max_workers=n_thread)
            for i in range(n_thread):
                if i < n_heavy:
                    # prcoess more chunk
                    n = n_per_heavy_thread
                    start = i * n_per_heavy_thread
                else:
                    # process less chunk
                    n = n_per_light_thread
                    start = n_heavy * n_per_heavy_thread + (i - n_heavy) * n_per_light_thread

                tasks.append(executor.submit(_thread_process, start, n))

            # wait for first exception or all completed
            done, _ = futures.wait(tasks, return_when=futures.FIRST_EXCEPTION)

            # raise from failed task
            try:
                for task in done:
                    task.result()
            except Exception as exc:
                # set err flag to notice running futures to terminate immediatly
                err = True
                executor.shutdown(wait=False)
                write_msg({"event": "complete", "oid": oid, "error": {"code": 2, "message": str(exc)}})
                raise exc
            else:
                executor.shutdown(wait=True)

            parts = sorted(parts, key=lambda part: part["partNumber"])

            r = get_session().post(
                completion_url,
                json={
                    "oid": oid,
                    "parts": parts,
                },
            )
            hf_raise_for_status(r)

            write_msg({"event": "complete", "oid": oid})
