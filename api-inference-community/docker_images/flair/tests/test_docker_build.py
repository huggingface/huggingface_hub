import os
import subprocess
from unittest import TestCase


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class DockerBuildTestCase(TestCase):
    def test_can_build_docker_image(self):
        with cd(os.path.dirname(os.path.dirname(__file__))):
            subprocess.check_output(["docker", "build", "."])
