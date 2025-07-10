import re
from argparse import Namespace, _SubParsersAction
from typing import Dict, Optional

import requests

from huggingface_hub import whoami
from huggingface_hub.utils import build_hf_headers

from .. import BaseHuggingfaceCLICommand
from ._cli_utils import tabulate


class PsCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("ps", help="List Jobs")
        run_parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Show all Jobs (default shows just running)",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        # Add Docker-style filtering argument
        run_parser.add_argument(
            "-f",
            "--filter",
            action="append",
            default=[],
            help="Filter output based on conditions provided (format: key=value)",
        )
        # Add option to format output
        run_parser.add_argument(
            "--format",
            type=str,
            help="Format output using a custom template",
        )
        run_parser.set_defaults(func=PsCommand)

    def __init__(self, args: Namespace) -> None:
        self.all: bool = args.all
        self.token: Optional[str] = args.token or None
        self.format: Optional[str] = args.format
        self.filters: Dict[str, str] = {}

        # Parse filter arguments (key=value pairs)
        for f in args.filter:
            if "=" in f:
                key, value = f.split("=", 1)
                self.filters[key.lower()] = value
            else:
                print(f"Warning: Ignoring invalid filter format '{f}'. Use key=value format.")

    def run(self) -> None:
        """
        Fetch and display job information for the current user.
        Uses Docker-style filtering with -f/--filter flag and key=value pairs.
        """
        try:
            # Get current username
            username = whoami(self.token)["name"]
            # Build headers for API request
            headers = build_hf_headers(token=self.token, library_name="hfjobs")
            # Fetch jobs data
            response = requests.get(
                f"https://huggingface.co/api/jobs/{username}",
                headers=headers,
                timeout=30,  # Add timeout to prevent hanging
            )
            response.raise_for_status()

            # Define table headers
            table_headers = ["JOB ID", "IMAGE/SPACE", "COMMAND", "CREATED", "STATUS"]

            # Process jobs data
            rows = []
            jobs = response.json()

            for job in jobs:
                # Extract job data for filtering
                status = job.get("status", {}).get("stage", "UNKNOWN")

                # Skip job if not all jobs should be shown and status doesn't match criteria
                if not self.all and status not in ("RUNNING", "UPDATING"):
                    continue

                # Extract job ID safely
                job_id = job.get("id", "N/A")

                # Extract image or space information
                if "spaceId" in job and job["spaceId"] is not None:
                    image_or_space = f"hf.co/spaces/{job['spaceId']}"
                else:
                    image_or_space = job.get("dockerImage", "N/A")

                # Extract and format command
                command = job.get("command", [])
                command_str = " ".join(command) if command else "N/A"

                # Extract creation time
                created_at = job.get("createdAt", "N/A")

                # Create a dict with all job properties for filtering
                job_properties = {
                    "id": job_id,
                    "image": image_or_space,
                    "status": status.lower(),
                    "command": command_str,
                }

                # Check if job matches all filters
                if not self._matches_filters(job_properties):
                    continue

                # Create row
                rows.append([job_id, image_or_space, command_str, created_at, status])

            # Handle empty results
            if not rows:
                filters_msg = ""
                if self.filters:
                    filters_msg = f" matching filters: {', '.join([f'{k}={v}' for k, v in self.filters.items()])}"

                print(f"No jobs found{filters_msg}")
                return

            # Apply custom format if provided or use default tabular format
            self._print_output(rows, table_headers)

        except requests.RequestException as e:
            print(f"Error fetching jobs data: {e}")
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing jobs data: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def _matches_filters(self, job_properties: Dict[str, str]) -> bool:
        """Check if job matches all specified filters."""
        for key, pattern in self.filters.items():
            # Check if property exists
            if key not in job_properties:
                return False

            # Support pattern matching with wildcards
            if "*" in pattern or "?" in pattern:
                # Convert glob pattern to regex
                regex_pattern = pattern.replace("*", ".*").replace("?", ".")
                if not re.search(f"^{regex_pattern}$", job_properties[key], re.IGNORECASE):
                    return False
            # Simple substring matching
            elif pattern.lower() not in job_properties[key].lower():
                return False

        return True

    def _print_output(self, rows, headers):
        """Print output according to the chosen format."""
        if self.format:
            # Custom template formatting (simplified)
            template = self.format
            for row in rows:
                line = template
                for i, field in enumerate(["id", "image", "command", "created", "status"]):
                    placeholder = f"{{{{.{field}}}}}"
                    if placeholder in line:
                        line = line.replace(placeholder, str(row[i]))
                print(line)
        else:
            # Default tabular format
            print(
                tabulate(
                    rows,
                    headers=headers,
                )
            )
