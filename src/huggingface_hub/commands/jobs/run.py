import io
import json
import time
from argparse import Namespace, _SubParsersAction
from typing import Optional, Union

import requests
from dotenv import dotenv_values

from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from .. import BaseHuggingfaceCLICommand


def _parse_timeout(timeout: Optional[str]) -> Optional[int]:
    """Get timeout in seconds"""
    time_units_factors = {"s": 1, "m": 60, "h": 3600, "d": 3600 * 24}
    if not timeout:
        return None
    elif timeout[-1] in time_units_factors:
        return int(float(timeout[:-1]) * time_units_factors[timeout[-1]])
    else:
        return int(timeout)


class RunCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("run", help="Run a Job")
        run_parser.add_argument("dockerImage", type=str, help="The Docker image to use.")
        run_parser.add_argument("-e", "--env", action="append", help="Set environment variables.")
        run_parser.add_argument("-s", "--secret", action="append", help="Set secret environment variables.")
        run_parser.add_argument("--env-file", type=str, help="Read in a file of environment variables.")
        run_parser.add_argument("--secret-env-file", type=str, help="Read in a file of secret environment variables.")
        run_parser.add_argument(
            "--flavor",
            type=str,
            help="Flavor for the hardware, as in HF Spaces.",
            default="cpu-basic",
        )
        run_parser.add_argument(
            "--timeout",
            type=str,
            help="Max duration: int/float with s (seconds, default), m (minutes), h (hours) or d (days).",
        )
        run_parser.add_argument(
            "-d",
            "--detach",
            action="store_true",
            help="Run the Job in the background and print the Job ID.",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        run_parser.add_argument("command", nargs="...", help="The command to run.")
        run_parser.set_defaults(func=RunCommand)

    def __init__(self, args: Namespace) -> None:
        self.docker_image: str = args.dockerImage
        self.environment: dict[str, Optional[str]] = {}
        for env_value in args.env or []:
            self.environment.update(dotenv_values(stream=io.StringIO(env_value)))
        if args.env_file:
            self.environment.update(dotenv_values(args.env_file))
        self.secrets: dict[str, Optional[str]] = {}
        for secret in args.secret or []:
            self.secrets.update(dotenv_values(stream=io.StringIO(secret)))
        if args.secret_env_file:
            self.secrets.update(dotenv_values(args.secret_env_file))
        self.flavor: str = args.flavor
        self.timeout: Optional[int] = _parse_timeout(args.timeout)
        self.detach: bool = args.detach
        self.token: Optional[str] = args.token
        self.command: list[str] = args.command

    def run(self) -> None:
        # prepare paypload to send to HF Jobs API
        input_json: dict[str, Optional[Union[str, float, list[str], dict[str, Optional[str]]]]] = {
            "command": self.command,
            "arguments": [],
            "environment": self.environment,
            "flavor": self.flavor,
        }
        # secrets are optional
        if self.secrets:
            input_json["secrets"] = self.secrets
        # timeout is optional
        if self.timeout:
            input_json["timeoutSeconds"] = self.timeout
        # input is either from docker hub or from HF spaces
        for prefix in (
            "https://huggingface.co/spaces/",
            "https://hf.co/spaces/",
            "huggingface.co/spaces/",
            "hf.co/spaces/",
        ):
            if self.docker_image.startswith(prefix):
                input_json["spaceId"] = self.docker_image[len(prefix) :]
                break
        else:
            input_json["dockerImage"] = self.docker_image
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        resp = requests.post(
            f"https://huggingface.co/api/jobs/{username}",
            json=input_json,
            headers=headers,
        )
        resp.raise_for_status()
        response = resp.json()
        # Fix: Update job_id extraction to match new response format
        job_id = response["id"]

        # Always print the job ID to the user
        print(f"Job started with ID: {job_id}")
        print(f"View at: https://huggingface.co/jobs/{username}/{job_id}")

        if self.detach:
            return

        # Now let's stream the logs

        logging_finished = logging_started = False
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
                    f"https://huggingface.co/api/jobs/{username}/{job_id}/logs",
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
                f"https://huggingface.co/api/jobs/{username}/{job_id}",
                headers=headers,
            ).json()
            if "status" in job_status and job_status["status"]["stage"] not in ("RUNNING", "UPDATING"):
                job_finished = True
            time.sleep(1)
