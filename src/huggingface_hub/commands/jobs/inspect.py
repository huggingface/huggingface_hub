import json
from argparse import Namespace, _SubParsersAction
from typing import Optional

import requests

from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from .. import BaseHuggingfaceCLICommand


class InspectCommand(BaseHuggingfaceCLICommand):

    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("inspect", help="Display detailed information on one or more Jobs")
        run_parser.add_argument(
            "--token", type=str, help="A User Access Token generated from https://huggingface.co/settings/tokens"
        )
        run_parser.add_argument(
            "jobs", nargs="...", help="The jobs to inspect"
        )
        run_parser.set_defaults(func=InspectCommand)

    def __init__(self, args: Namespace) -> None:
        self.token: Optional[str] = args.token or None
        self.jobs: list[str] = args.jobs

    def run(self) -> None:
        username = whoami(self.token)["name"]
        headers = build_hf_headers(token=self.token, library_name="hfjobs")
        inspections = [
            requests.get(
                f"https://huggingface.co/api/jobs/{username}/{job}",
                headers=headers,
            ).json()
            for job in self.jobs
        ]
        print(json.dumps(inspections, indent=4))
