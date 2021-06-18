import json
import os
import subprocess
import time
import unittest
import uuid
from collections import Counter

import httpx


class DockerPopen(subprocess.Popen):
    def __exit__(self, exc_type, exc_val, traceback):
        self.terminate()
        self.wait(5)
        return super().__exit__(exc_type, exc_val, traceback)


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


@unittest.skipIf(
    "RUN_DOCKER_TESTS" not in os.environ,
    "Docker tests are slow, set `RUN_DOCKER_TESTS=1` environement variable to run them",
)
class DockerImageTests(unittest.TestCase):
    def create_docker(self, name: str) -> str:
        rand = str(uuid.uuid4())[:5]
        tag = f"{name}:{rand}"
        with cd(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "docker_images", name
            )
        ):
            proc = subprocess.run(["docker", "build", ".", "-t", tag])
        self.assertEqual(proc.returncode, 0)
        return tag

    def test_allennlp(self):
        self.framework_docker_test(
            "allennlp", "question-answering", "lysandre/bidaf-elmo-model-2020.03.19"
        )
        self.framework_invalid_test("allennlp")

    def test_asteroid(self):
        self.framework_docker_test(
            "asteroid",
            "audio-source-separation",
            "mhu-coder/ConvTasNet_Libri1Mix_enhsingle",
        )
        self.framework_docker_test(
            "asteroid",
            "audio-source-separation",
            "julien-c/DPRNNTasNet-ks16_WHAM_sepclean",
        )
        self.framework_invalid_test("asteroid")

    def test_espnet(self):
        self.framework_docker_test(
            "espnet",
            "text-to-speech",
            "julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train",
        )
        self.framework_invalid_test("espnet")
        # TOO SLOW
        # (
        #     "espnet",
        #     "automatic-speech-recognition",
        #     "julien-c/mini_an4_asr_train_raw_bpe_valid",
        # ),

    def test_sentence_transformers(self):
        self.framework_docker_test(
            "sentence_transformers",
            "feature-extraction",
            "bert-base-uncased",
        )

        self.framework_docker_test(
            "sentence_transformers",
            "sentence-similarity",
            "paraphrase-distilroberta-base-v1",
        )
        self.framework_invalid_test("sentence_transformers")

    def test_flair(self):
        self.framework_docker_test(
            "flair", "token-classification", "flair/chunk-english-fast"
        )
        self.framework_invalid_test("flair")

    def test_spacy(self):
        self.framework_docker_test(
            "spacy",
            "token-classification",
            "spacy/en_core_web_sm",
        )
        self.framework_invalid_test("spacy")

    def test_speechbrain(self):
        self.framework_docker_test(
            "speechbrain",
            "automatic-speech-recognition",
            "speechbrain/asr-crdnn-commonvoice-it",
        )
        self.framework_invalid_test("speechbrain")

    def test_timm(self):
        self.framework_docker_test("timm", "image-classification", "sgugger/resnet50d")
        self.framework_invalid_test("timm")

    def framework_invalid_test(self, framework: str):
        task = "invalid"
        model_id = "invalid"
        tag = self.create_docker(framework)
        run_docker_command = [
            "docker",
            "run",
            "-p",
            "8000:80",
            "-e",
            f"TASK={task}",
            "-e",
            f"MODEL_ID={model_id}",
            "-v",
            "/tmp:/data",
            "-t",
            tag,
        ]

        url = "http://localhost:8000"
        timeout = 60
        with DockerPopen(run_docker_command) as proc:
            for i in range(400):
                try:
                    response = httpx.get(url, timeout=10)
                    break
                except Exception:
                    time.sleep(1)
            self.assertEqual(response.content, b'{"ok":"ok"}')

            response = httpx.post(url, data=b"This is a test", timeout=timeout)
            self.assertIn(response.status_code, {400, 500})
            self.assertEqual(response.headers["content-type"], "application/json")

            proc.terminate()
            proc.wait(5)

    def framework_docker_test(self, framework: str, task: str, model_id: str):
        tag = self.create_docker(framework)
        run_docker_command = [
            "docker",
            "run",
            "-p",
            "8000:80",
            "-e",
            f"TASK={task}",
            "-e",
            f"MODEL_ID={model_id}",
            "-v",
            "/tmp:/data",
            "-t",
            tag,
        ]

        url = "http://localhost:8000"
        timeout = 60
        counter = Counter()
        with DockerPopen(run_docker_command) as proc:
            for i in range(400):
                try:
                    response = httpx.get(url, timeout=10)
                    break
                except Exception:
                    time.sleep(1)
            self.assertEqual(response.content, b'{"ok":"ok"}')

            response = httpx.post(url, data=b"This is a test", timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={"inputs": "This is a test"},
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={
                    "inputs": {"question": "This is a test", "context": "Some context"}
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            response = httpx.post(
                url,
                json={
                    "inputs": {
                        "source_sentence": "This is a test",
                        "sentences": ["Some context", "Something else"],
                    }
                },
                timeout=timeout,
            )
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "sample1.flac"), "rb"
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1
            if response.status_code == 200:
                if response.headers["content-type"] == "application/json":
                    data = json.loads(response.content)
                    self.assertEqual(set(data.keys()), {"text"})
                elif response.headers["content-type"] == "audio/flac":
                    pass
                else:
                    raise Exception("Unknown format")

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "malformed.flac"),
                "rb",
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "plane.jpg"), "rb"
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "sample1_dual.ogg"),
                "rb",
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            with open(
                os.path.join(os.path.dirname(__file__), "samples", "sample1.webm"), "rb"
            ) as f:
                data = f.read()
            response = httpx.post(url, data=data, timeout=timeout)
            self.assertIn(response.status_code, {200, 400})
            counter[response.status_code] += 1

            proc.terminate()
            proc.wait(5)

        self.assertEqual(proc.returncode, 0)
        self.assertGreater(
            counter[200],
            0,
            f"At least one request should have gone through {framework}, {task}, {model_id}",
        )

        # Follow up loading are much faster, 20s should be ok.
        with DockerPopen(run_docker_command) as proc2:
            for i in range(20):
                try:
                    response = httpx.get(url, timeout=10)
                    break
                except Exception:
                    time.sleep(1)
            self.assertEqual(response.content, b'{"ok":"ok"}')
            proc2.terminate()
            proc2.wait(5)
        self.assertEqual(proc2.returncode, 0)
