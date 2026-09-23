"""Offline regression tests for the reusable style bot's actual YAML steps.

Run with Python, PyYAML, Node, Bash and Git installed. No GitHub credentials or
GitHub API calls are replaced with local fixtures. Set STYLE_BOT_DOCKER_TESTS=1
to build the tooling image and exercise the actual container isolation as well.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
WORKFLOW = yaml.safe_load((WORKFLOWS / "style-bot-action.yml").read_text())
REPOSITORY = WORKFLOWS.parents[1]
SHA = "a" * 40


def step(job, name):
    return next(item for item in WORKFLOW["jobs"][job]["steps"] if item["name"] == name)


class StyleBotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = os.environ.get("STYLE_BOT_TEST_IMAGE")
        if not cls.image and os.environ.get("STYLE_BOT_DOCKER_TESTS") == "1":
            with tempfile.TemporaryDirectory(prefix="style-bot-build-") as temporary:
                output = Path(temporary) / "outputs"
                result = subprocess.run(
                    [
                        "bash",
                        "-e",
                        "-o",
                        "pipefail",
                        "-c",
                        step("run-style-bot", "Build tooling from the base repository")["run"],
                    ],
                    cwd=REPOSITORY,
                    env=os.environ
                    | {
                        "PYTHON_VERSION": "3.10",
                        "PYTHON_QUALITY_DEPENDENCIES": "[quality]",
                        "GITHUB_OUTPUT": str(output),
                    },
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                cls.image = output.read_text().strip().removeprefix("image_id=")
                cls.addClassCleanup(subprocess.run, ["docker", "image", "rm", cls.image], capture_output=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="style-bot-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "workspace"
        self.root.mkdir()
        self.pr = self.root / "pr-repo"
        self.output = Path(self.temporary.name) / "outputs"
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": self.temporary.name,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Style bot test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Style bot test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GITHUB_WORKSPACE": str(self.root),
            "GITHUB_OUTPUT": str(self.output),
            "GITHUB_SERVER_URL": "https://github.example.invalid",
            "GITHUB_REPOSITORY": "base/repo",
            "GITHUB_RUN_ID": "123",
        }
        if "DOCKER_HOST" in os.environ:
            self.environment["DOCKER_HOST"] = os.environ["DOCKER_HOST"]
        if self.image:
            self.environment["STYLE_BOT_IMAGE"] = self.image
        self.git(self.root, "init", "-q", "-b", "main")
        (self.root / ".gitignore").write_text("pr-repo/\n")
        for name in ("setup.py", "setup.cfg", "pyproject.toml"):
            (self.root / name).write_text("# trusted configuration\n")
        (self.root / "utils").mkdir()
        (self.root / "utils" / "check.py").write_text("# trusted utility\n")
        (self.root / "Makefile").write_text("style:\n\t@true\nquality:\n\t@true\n")
        self.commit(self.root)
        self.git(self.root, "clone", "-q", "--no-hardlinks", str(self.root), str(self.pr))

    def git(self, directory, *args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], env=self.environment, text=True, stderr=subprocess.PIPE
        ).strip()

    def commit(self, directory):
        self.git(directory, "add", "-A")
        self.git(directory, "commit", "-qm", "test fixture")
        return self.git(directory, "rev-parse", "HEAD")

    def shell(self, job, name, **environment):
        self.output.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step(job, name)["run"]],
            cwd=self.root,
            env=self.environment | environment,
            capture_output=True,
            text=True,
        )
        outputs = self.output.read_text() if self.output.exists() else ""
        return result, outputs

    def javascript(self, job, name, *, body=None, permission="write", pr=None, **environment):
        payload = {"issue": {"number": 17}, "comment": {"body": body, "user": {"login": "maintainer"}}}
        fixture = {"permission": permission, "pr": pr, "payload": payload}
        script = (
            """
const fixture = JSON.parse(process.env.FIXTURE);
const result = {outputs: {}, errors: [], comments: [], requests: []};
const core = {setOutput: (k,v) => result.outputs[k] = v, setFailed: v => result.errors.push(v)};
const console = {log: () => {}};
const context = {repo: {owner: 'base', repo: 'repo'}, payload: fixture.payload};
const github = {rest: {
  repos: {getCollaboratorPermissionLevel: async () => ({data: {permission: fixture.permission}})},
  pulls: {get: async args => {result.requests.push(args); return {data: fixture.pr};}},
  issues: {
    createComment: async args => {result.comments.push(args); return {data: {id: 42}};},
    updateComment: async args => {result.comments.push(args);},
  },
}};
(async () => {
"""
            + step(job, name)["with"]["script"]
            + "\n})().then(() => process.stdout.write(JSON.stringify(result)));"
        )
        result = subprocess.run(
            ["node", "-e", script],
            env=self.environment | environment | {"FIXTURE": json.dumps(fixture)},
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_only_an_authorized_exact_sha_command_is_accepted(self):
        for body, permission, expected in [
            (f"@bot /style {SHA}", "write", SHA),
            (f"@bot /style {SHA.upper()}\n", "maintain", SHA),
            (f"@bot /style\t{SHA}", "admin", SHA),
            (f"@bot /style {SHA}", "read", None),
            ("@bot /style", "write", None),
            (f"@bot /style {SHA[:7]}", "write", None),
            (f"@bot /style {SHA}\nextra text", "write", None),
            (f"@bot /style\n{SHA}", "write", None),
            (f"@bot /stylesheet {SHA}", "write", None),
            (f"@bot /style {SHA}; false", "write", None),
        ]:
            with self.subTest(body=body, permission=permission):
                result = self.javascript(
                    "check-permissions", "Check user permission", body=body, permission=permission
                )
                self.assertEqual(result["outputs"].get("approvedSha"), expected)

    def test_api_cannot_replace_the_authorized_sha(self):
        for job, name in [
            ("run-style-bot", "Validate the authorized commit against the PR"),
            ("push-style-fixes", "Validate the PR before publishing"),
        ]:
            for head, state, repo in [
                (SHA, "open", {"full_name": "contributor/fork"}),
                ("b" * 40, "open", {"full_name": "contributor/fork"}),
                (SHA, "closed", {"full_name": "contributor/fork"}),
                (SHA, "open", None),
            ]:
                with self.subTest(job=job, head=head, state=state, repo=repo):
                    pr = {"state": state, "head": {"sha": head, "repo": repo, "ref": "contribution"}}
                    result = self.javascript(job, name, pr=pr, APPROVED_SHA=SHA)
                    self.assertEqual(result["requests"][0]["pull_number"], 17)
                    if head == SHA and state == "open" and repo:
                        self.assertFalse(result["errors"])
                        self.assertEqual(result["outputs"]["headRepoFullName"], "contributor/fork")
                    else:
                        self.assertTrue(result["errors"])
                        self.assertNotIn("headRepoFullName", result["outputs"])

    def test_publisher_destination_does_not_come_from_worker_outputs(self):
        publisher = WORKFLOW["jobs"]["push-style-fixes"]
        checkout = step("push-style-fixes", "Check out the reviewed commit")["with"]
        self.assertEqual(checkout["ref"], "${{ needs.check-permissions.outputs.approvedSha }}")
        self.assertEqual(checkout["repository"], "${{ github.repository }}")
        names = [item["name"] for item in publisher["steps"]]
        self.assertLess(names.index("Validate and apply style fixes"), names.index("Generate bot token"))
        for item in publisher["steps"]:
            self.assertNotIn("needs.run-style-bot.outputs", json.dumps(item))

    def test_protected_tree_identity_includes_deletions_renames_and_modes(self):
        original = self.git(self.pr, "rev-parse", "HEAD")
        for mutation in ("unchanged", "edit", "delete", "rename", "mode", "symlink", "directory_symlink", "setup"):
            with self.subTest(mutation=mutation):
                self.git(self.pr, "reset", "--hard", original)
                self.git(self.pr, "clean", "-fd")
                utility = self.pr / "utils" / "check.py"
                if mutation == "edit":
                    utility.write_text("# changed utility\n")
                elif mutation == "delete":
                    utility.unlink()
                elif mutation == "rename":
                    utility.rename(self.pr / "renamed.py")
                elif mutation == "mode":
                    utility.chmod(0o755)
                elif mutation == "symlink":
                    utility.unlink()
                    utility.symlink_to("../setup.py")
                elif mutation == "directory_symlink":
                    shutil.rmtree(self.pr / "utils")
                    (self.pr / "utils").symlink_to("../utils")
                elif mutation == "setup":
                    (self.pr / "setup.py").write_text("# changed setup\n")
                if mutation != "unchanged":
                    self.commit(self.pr)
                result, outputs = self.shell("run-style-bot", "Refuse pull requests that modify tooling entrypoints")
                self.assertEqual(result.returncode == 0, mutation == "unchanged", result.stderr)
                self.assertIn(f"entrypoints_modified={str(mutation != 'unchanged').lower()}", outputs)

    def test_later_clean_commit_does_not_hide_modified_approved_utils(self):
        utility = self.pr / "utils" / "check.py"
        utility.write_text("# changed approved utility\n")
        approved = self.commit(self.pr)
        utility.write_text("# trusted utility\n")
        later = self.commit(self.pr)
        self.assertEqual(self.git(self.pr, "diff", "--name-only", "HEAD~2", later), "")
        self.git(self.pr, "checkout", "-q", "--detach", approved)
        result, outputs = self.shell("run-style-bot", "Refuse pull requests that modify tooling entrypoints")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("entrypoints_modified=true", outputs)

    def test_all_modes_propagate_failures_and_use_the_trusted_makefile(self):
        if not self.image:
            self.skipTest("Set STYLE_BOT_DOCKER_TESTS=1 to test the real container")
        (self.pr / "Makefile").write_text("$(error Contributor Makefile must not run)\n")
        self.commit(self.pr)
        for mode in ("default", "style_only", "quality_only"):
            for failing in ("none", "style", "quality"):
                with self.subTest(mode=mode, failing=failing):
                    visited = self.pr / "visited"
                    visited.unlink(missing_ok=True)
                    makefile = ""
                    for target in ("style", "quality"):
                        makefile += f"{target}:\n\t@echo {target} >> visited\n"
                        if target == failing:
                            makefile += "\t@false\n"
                    (self.root / "Makefile").write_text(makefile)
                    result, outputs = self.shell("run-style-bot", "Run style command", STYLECOMMANDTYPE=mode)
                    expected = ["style", "quality"] if mode == "default" else [mode.removesuffix("_only")]
                    failure_expected = failing in expected
                    self.assertEqual(result.returncode != 0, failure_expected, result.stderr)
                    if failure_expected:
                        expected = expected[: expected.index(failing) + 1]
                        self.assertNotIn("changes_detected=true", outputs)
                    else:
                        self.assertIn("changes_detected=true", outputs)
                    self.assertEqual(visited.read_text().splitlines(), expected)
                    self.assertEqual(list(self.pr.glob(".style-bot.Makefile.*")), [])

    def test_clean_tree_and_existing_makefile_symlink(self):
        if not self.image:
            self.skipTest("Set STYLE_BOT_DOCKER_TESTS=1 to test the real container")
        (self.pr / ".style-bot.Makefile").symlink_to("utils/check.py")
        self.commit(self.pr)
        before = (self.pr / "utils" / "check.py").read_text()
        result, outputs = self.shell("run-style-bot", "Run style command", STYLECOMMANDTYPE="default")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("changes_detected=false", outputs)
        self.assertEqual((self.pr / "utils" / "check.py").read_text(), before)
        self.assertTrue((self.pr / ".style-bot.Makefile").is_symlink())

    def test_invalid_command_gets_usage_instead_of_authorizing_a_run(self):
        result = self.javascript("init-comment", "Comment on PR with workflow run link", APPROVED_SHA="")
        self.assertIn("full 40-character commit SHA", result["comments"][0]["body"])
        self.assertEqual(result["comments"][0]["issue_number"], 17)

    def test_push_lease_rejects_a_move_after_the_api_check(self):
        approved = self.git(self.root, "rev-parse", "HEAD")
        for branch_moved in (False, True):
            with self.subTest(branch_moved=branch_moved):
                self.git(self.root, "reset", "--hard", approved)
                remote = Path(self.temporary.name) / f"remote-{branch_moved}.git"
                self.git(self.root, "clone", "-q", "--bare", str(self.root), str(remote))
                if "origin" not in self.git(self.root, "remote").splitlines():
                    self.git(self.root, "remote", "add", "origin", str(remote))
                # Exercise the exact publish block without contacting GitHub.
                # No network protocol is allowed, and the credential is synthetic.
                remote_url = f"https://x-access-token:dummy@github.com/fixture/fork-{branch_moved}.git"
                self.git(self.root, "config", f"url.{remote.as_uri()}.insteadOf", remote_url)
                artifact = Path(self.temporary.name) / f"artifact-{branch_moved}"
                artifact.mkdir()
                (self.root / "formatted.txt").write_text("style fix\n")
                self.git(self.root, "add", "formatted.txt")
                patch = subprocess.check_output(
                    ["git", "-C", str(self.root), "diff", "--cached", "--binary"], env=self.environment
                )
                (artifact / "style-fixes.patch").write_bytes(patch)
                self.git(self.root, "reset", "--hard", approved)
                if branch_moved:
                    (self.pr / "later.txt").write_text("later contributor commit\n")
                    later = self.commit(self.pr)
                    self.git(self.pr, "push", "-q", str(remote), "HEAD:refs/heads/main")
                result, _ = self.shell(
                    "push-style-fixes",
                    "Validate and apply style fixes",
                    STYLE_BOT_OUTPUT_DIR=str(artifact),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                result, _ = self.shell(
                    "push-style-fixes",
                    "Push validated style fixes",
                    HEADREPOFULLNAME=f"fixture/fork-{branch_moved}",
                    HEADREF="main",
                    APPROVED_SHA=approved,
                    GITHUB_TOKEN="dummy",
                    GIT_ALLOW_PROTOCOL="file",
                )
                self.assertEqual(result.returncode != 0, branch_moved, result.stderr)
                actual = self.git(remote, "rev-parse", "refs/heads/main")
                if branch_moved:
                    self.assertEqual(actual, later)
                else:
                    self.assertEqual(self.git(remote, "show", "main:formatted.txt"), "style fix")
                    self.assertEqual(self.git(remote, "rev-parse", "main^"), approved)

    def test_publisher_rejects_protected_paths_and_symlinks_in_worker_patch(self):
        original = self.git(self.root, "rev-parse", "HEAD")
        artifact = Path(self.temporary.name) / "patch"
        artifact.mkdir()
        for name, symlink in [
            ("src/normal.py", False),
            (".github/workflows/injected.yml", False),
            (".gitattributes", False),
            ("utils/check.py", False),
            ("setup.py", False),
            ("Makefile", False),
            ("src/link", True),
        ]:
            with self.subTest(name=name):
                self.git(self.root, "reset", "--hard", original)
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if symlink:
                    target.symlink_to("/tmp/outside")
                else:
                    target.write_text("# worker output\n")
                self.git(self.root, "add", "--", name)
                patch = subprocess.check_output(
                    ["git", "-C", str(self.root), "diff", "--cached", "--binary"], env=self.environment
                )
                (artifact / "style-fixes.patch").write_bytes(patch)
                self.git(self.root, "reset", "--hard", original)
                result, _ = self.shell(
                    "push-style-fixes", "Validate and apply style fixes", STYLE_BOT_OUTPUT_DIR=str(artifact)
                )
                self.assertEqual(result.returncode == 0, name == "src/normal.py", result.stderr)

    def test_imported_pr_code_cannot_access_runner_or_modify_tooling(self):
        if not self.image:
            self.skipTest("Set STYLE_BOT_DOCKER_TESTS=1 to test the real container")
        marker = Path(self.temporary.name) / "runner-only-marker"
        marker.write_text("inert runner marker")
        script = """
import json
import os
import socket
from pathlib import Path

blocked = []
for filename in ('.git/config', '/opt/style-bot/Makefile', '/opt/style-bot/source/setup.py'):
    try:
        with open(filename, 'a') as stream:
            stream.write('inert marker')
    except OSError:
        blocked.append(filename)
try:
    socket.create_connection(('192.0.2.1', 80), timeout=1)
    network_blocked = False
except OSError:
    network_blocked = True
Path('container-result.json').write_text(json.dumps({
    'blocked': blocked,
    'network_blocked': network_blocked,
    'uid': os.getuid(),
    'interfaces_up': sorted(
        path.name for path in Path('/sys/class/net').iterdir()
        if path.is_dir() and int((path / 'flags').read_text(), 16) & 1
    ),
    'runner_env': sorted(key for key in os.environ if key.startswith(('GITHUB_', 'ACTIONS_', 'RUNNER_'))),
    'docker_socket': Path('/var/run/docker.sock').exists(),
    'process_status': Path('/proc/self/status').read_text(),
    'runner_marker': Path(RUNNER_MARKER).exists(),
}))
""".replace("RUNNER_MARKER", repr(str(marker)))
        (self.pr / "src").mkdir()
        (self.pr / "src" / "canary.py").write_text(script)
        (self.root / "Makefile").write_text('style:\n\tpython -c "import canary"\nquality:\n\t@true\n')
        self.commit(self.pr)
        before = (self.pr / ".git/config").read_bytes()
        result, outputs = self.shell("run-style-bot", "Run style command", STYLECOMMANDTYPE="default")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        proof = json.loads((self.pr / "container-result.json").read_text())
        self.assertEqual(len(proof["blocked"]), 3)
        self.assertTrue(proof["network_blocked"])
        self.assertNotEqual(proof["uid"], 0)
        self.assertEqual(proof["interfaces_up"], ["lo"])
        self.assertEqual(proof["runner_env"], [])
        self.assertFalse(proof["docker_socket"])
        self.assertFalse(proof["runner_marker"])
        self.assertIn("CapEff:\t0000000000000000", proof["process_status"])
        self.assertIn("NoNewPrivs:\t1", proof["process_status"])
        self.assertEqual((self.pr / ".git/config").read_bytes(), before)
        self.assertEqual(marker.read_text(), "inert runner marker")
        self.assertIn("changes_detected=true", outputs)

    def test_real_hub_tree_in_all_three_container_modes(self):
        if not self.image:
            self.skipTest("Set STYLE_BOT_DOCKER_TESTS=1 to test the real container")
        archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=REPOSITORY)
        subprocess.run(["tar", "-x", "-C", str(self.pr)], input=archive, check=True)
        shutil.copyfile(REPOSITORY / "Makefile", self.root / "Makefile")
        original = self.commit(self.pr)
        for mode in ("style_only", "default", "quality_only"):
            with self.subTest(mode=mode):
                self.git(self.pr, "reset", "--hard", original)
                if mode != "quality_only":
                    (self.pr / "src/huggingface_hub/_style_bot_smoke.py").write_text(
                        "def add(left: int,right:int)->int:\n return left+right\n"
                    )
                    self.commit(self.pr)
                result, outputs = self.shell("run-style-bot", "Run style command", STYLECOMMANDTYPE=mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"changes_detected={str(mode != 'quality_only').lower()}", outputs)
                if mode != "quality_only":
                    self.assertEqual(
                        (self.pr / "src/huggingface_hub/_style_bot_smoke.py").read_text(),
                        "def add(left: int, right: int) -> int:\n    return left + right\n",
                    )

    def test_reporting_requires_successful_publication_and_keeps_the_run_link(self):
        cases = [
            ({"PUSH_RESULT": "success"}, "pushed the changes"),
            ({"PUSH_RESULT": "failure"}, "publication did not complete successfully"),
            ({"PUSH_RESULT": "skipped"}, "publication did not complete successfully"),
            ({"PUSH_RESULT": "cancelled"}, "cancelled"),
            ({"STYLE_RESULT": "failure"}, "did not complete successfully"),
            ({"ENTRYPOINTS_MODIFIED": "true"}, "different `setup.py`"),
            ({"BRANCH_MOVED": "true", "PUSH_RESULT": "failure"}, "branch moved"),
            ({"CHANGES_DETECTED": "false", "PUSH_RESULT": "skipped"}, "without modifying any files"),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                environment = {"STYLE_RESULT": "success", "CHANGES_DETECTED": "true", "COMMENT_ID": "42"} | overrides
                result = self.javascript("finalize-comment", "Update the bot comment with the outcome", **environment)
                body = result["comments"][0]["body"]
                self.assertIn(expected, body)
                self.assertIn("https://github.example.invalid/base/repo/actions/runs/123", body)
                self.assertEqual("pushed the changes" in body, overrides.get("PUSH_RESULT") == "success")


class StaticMetadataTests(unittest.TestCase):
    def test_static_checks_do_not_import_the_pr_package(self):
        with tempfile.TemporaryDirectory(prefix="style-bot-static-") as temporary:
            root = Path(temporary)
            (root / "utils").mkdir()
            for name in ("helpers.py", "check_static_imports.py", "check_all_variable.py"):
                shutil.copyfile(REPOSITORY / "utils" / name, root / "utils" / name)
            shutil.copyfile(REPOSITORY / "pyproject.toml", root / "pyproject.toml")
            marker = root / "import-executed"
            for relative in ("src/huggingface_hub/__init__.py", "src/huggingface_hub/utils/__init__.py"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                content = (REPOSITORY / relative).read_text()
                boundary = "\nif TYPE_CHECKING:  # pragma: no cover\n"
                canary = f"\n__import__('pathlib').Path({str(marker)!r}).write_text('inert marker')\n"
                target.write_text(content.replace(boundary, canary + boundary))
            environment = {
                "PATH": os.environ["PATH"],
                "HOME": temporary,
                "PYTHONPATH": str(root / "src"),
            }
            for script in ("check_static_imports.py", "check_all_variable.py"):
                for arguments in (["--update"], []):
                    with self.subTest(script=script, arguments=arguments):
                        result = subprocess.run(
                            [sys.executable, str(root / "utils" / script), *arguments],
                            cwd=root,
                            env=environment,
                            text=True,
                            capture_output=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertFalse(marker.exists())

            # The mapping itself must be a literal. Evaluating a call to obtain
            # it would reintroduce execution even without an import statement.
            (root / "src/huggingface_hub/__init__.py").write_text(
                f"_SUBMOD_ATTRS = __import__('pathlib').Path({str(marker)!r}).write_text('inert marker')\n"
            )
            for script in ("check_static_imports.py", "check_all_variable.py"):
                with self.subTest(script=script, nonliteral=True):
                    result = subprocess.run(
                        [sys.executable, str(root / "utils" / script), "--update"],
                        cwd=root,
                        env=environment,
                        text=True,
                        capture_output=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
