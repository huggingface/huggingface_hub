import json
import time
from argparse import Namespace, _SubParsersAction
from typing import Optional

import requests

from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from .. import BaseHuggingfaceCLICommand


class LogsCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("logs", help="Fetch the logs of a Job")
        run_parser.add_argument("job_id", type=str, help="Job ID")
        run_parser.add_argument("-t", "--timestamps", action="store_true", help="Show timestamps")
        run_parser.add_argument(
            "--token", type=str, help="A User Access Token generated from https://huggingface.co/settings/tokens"
        )
        run_parser.set_defaults(func=LogsCommand)

    def __init__(self, args: Namespace) -> None:
        self.job_id: str = args.job_id
        self.timestamps: bool = args.timestamps
        self.token: Optional[str] = args.token or None

    def run(self) -> None:
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        requests.get(
            f"https://huggingface.co/api/jobs/{username}/{self.job_id}",
            headers=headers,
        ).raise_for_status()

        logging_started = False
        logging_finished = False
        job_finished = False
        # - We need to retry because sometimes the /logs doesn't return logs when the job just started.
        #   (for example it can return only two lines: one for "Job started" and one empty line)
        # - Timeouts can happen in case of build errors
        # - ChunkedEncodingError can happen in case of stopped logging in the middle of streaming
        # - Infinite empty log stream can happen in case of build error
        #   (the logs stream is infinite and empty except for the Job started message)
        # - there is a ": keep-alive" every 30 seconds
        while True:
            try:
                resp = requests.get(
                    f"https://huggingface.co/api/jobs/{username}/{self.job_id}/logs",
                    headers=headers,
                    stream=True,
                    timeout=120,
                )
                log = None
                for line in resp.iter_lines(chunk_size=1):
                    line = line.decode("utf-8")
                    if line and line.startswith("data: {"):
                        data = json.loads(line[len("data: ") :])
                        # timestamp = data["timestamp"]
                        if not data["data"].startswith("===== Job started"):
                            logging_started = True
                            log = data["data"]
                            print(log)
                logging_finished = logging_started
            except requests.exceptions.ChunkedEncodingError:
                # Response ended prematurely
                break
            except KeyboardInterrupt:
                break
            except requests.exceptions.ConnectionError as err:
                is_timeout = err.__context__ and isinstance(err.__context__.__cause__, TimeoutError)
                if logging_started or not is_timeout:
                    raise
            if logging_finished or job_finished:
                break
            job_status = requests.get(
                f"https://huggingface.co/api/jobs/{username}/{self.job_id}",
                headers=headers,
            ).json()
            if "status" in job_status and job_status["status"]["stage"] not in ("RUNNING", "UPDATING"):
                job_finished = True
            time.sleep(1)
