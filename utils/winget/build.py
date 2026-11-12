"""Build the Windows x86_64 portable distribution and its WinGet manifests.

Run on Windows from the repository root, after compiling launcher.c to dist/hf.exe.
The full Python runtime keeps subprocesses and Python extensions working without freezing.
"""

import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import yaml


PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
    "cpython-3.12.16%2B20261110-x86_64-pc-windows-msvc-install_only.tar.gz"
)
PYTHON_SHA256 = "e23fe5b16926de1d81661b1a97352e89fb167549c55a6825796ccd0f32ec7f1b"
PACKAGE_ID = "HuggingFace.hf"
ASSET_NAME = "hf-windows-x86_64.zip"
REPOSITORY = "https://github.com/huggingface/huggingface_hub"


def write_manifests(version: str, sha256: str, output: Path) -> Path:
    """Generate the three files required by the WinGet community repository."""
    destination = output / "manifests" / "h" / "HuggingFace" / "hf" / version
    destination.mkdir(parents=True, exist_ok=True)
    common = {"PackageIdentifier": PACKAGE_ID, "PackageVersion": version}
    manifests = {
        "": {"DefaultLocale": "en-US", "ManifestType": "version"},
        ".installer": {
            "InstallerType": "zip",
            "NestedInstallerType": "portable",
            "NestedInstallerFiles": [{"RelativeFilePath": "hf.exe", "PortableCommandAlias": "hf"}],
            "Commands": ["hf"],
            "Scope": "user",
            "Installers": [
                {
                    "Architecture": "x64",
                    "InstallerUrl": f"{REPOSITORY}/releases/download/v{version}/{ASSET_NAME}",
                    "InstallerSha256": sha256.upper(),
                }
            ],
            "ManifestType": "installer",
        },
        ".locale.en-US": {
            "PackageLocale": "en-US",
            "Publisher": "Hugging Face",
            "PublisherUrl": "https://huggingface.co",
            "PublisherSupportUrl": f"{REPOSITORY}/issues",
            "PackageName": "Hugging Face CLI",
            "PackageUrl": "https://huggingface.co/docs/huggingface_hub/guides/cli",
            "License": "Apache-2.0",
            "LicenseUrl": f"{REPOSITORY}/blob/v{version}/LICENSE",
            "ShortDescription": "The command-line interface for the Hugging Face Hub.",
            "Moniker": "hf",
            "Tags": ["huggingface", "machine-learning", "cli"],
            "ReleaseNotesUrl": f"{REPOSITORY}/releases/tag/v{version}",
            "ManifestType": "defaultLocale",
        },
    }
    for suffix, fields in manifests.items():
        manifest = {**common, **fields, "ManifestVersion": "1.9.0"}
        (destination / f"{PACKAGE_ID}{suffix}.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
    return destination


def main() -> None:
    output = Path("dist").resolve()
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        archive = staging / "python.tar.gz"
        urllib.request.urlretrieve(PYTHON_URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != PYTHON_SHA256:
            raise RuntimeError("Python runtime checksum mismatch")
        bundle = staging / "bundle"
        bundle.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(bundle, filter="data")
        python = bundle / "python" / "python.exe"
        subprocess.run([str(python), "-I", "-m", "ensurepip"], check=True)
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", "--no-compile", "--only-binary=:all:", "."], check=True
        )
        version = subprocess.check_output(
            [str(python), "-I", "-c", "import huggingface_hub; print(huggingface_hub.__version__)"], text=True
        ).strip()
        (bundle / "python" / ".hf_winget_marker").touch()
        shutil.copy2(output / "hf.exe", bundle / "hf.exe")
        shutil.copy2("LICENSE", bundle / "LICENSE")
        # Keep Python's and dependencies' licenses/metadata in the bundle.
        with (bundle / "requirements.txt").open("w", encoding="utf-8") as requirements:
            subprocess.run(
                [str(python), "-I", "-m", "pip", "list", "--format=freeze"], check=True, stdout=requirements
            )
        zip_path = Path(shutil.make_archive(str(output / ASSET_NAME.removesuffix(".zip")), "zip", bundle))
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (output / f"{ASSET_NAME}.sha256").write_text(f"{sha256}  {ASSET_NAME}\n", encoding="utf-8")
    write_manifests(version, sha256, output)
    shutil.make_archive(str(output / "hf-winget-manifests"), "zip", output, "manifests")


if __name__ == "__main__":
    main()
