from unittest import TestCase

import torch
from api_inference_community.normalizers import speaker_diarization_normalize


class NormalizersTestCase(TestCase):
    def test_speaker_diarization_dummy(self):
        tensor = torch.zeros((10, 2))
        outputs = speaker_diarization_normalize(
            tensor, 16000, ["SPEAKER_0", "SPEAKER_1"]
        )
        self.assertEqual(outputs, [])

    def test_speaker_diarization(self):
        tensor = torch.zeros((10, 2))
        tensor[1:4, 0] = 1
        tensor[3:8, 1] = 1
        tensor[8:10, 0] = 1
        outputs = speaker_diarization_normalize(
            tensor, 16000, ["SPEAKER_0", "SPEAKER_1"]
        )
        self.assertEqual(
            outputs,
            [
                {"class": "SPEAKER_0", "start": 1 / 16000, "end": 4 / 16000},
                {"class": "SPEAKER_1", "start": 3 / 16000, "end": 8 / 16000},
                {"class": "SPEAKER_0", "start": 8 / 16000, "end": 10 / 16000},
            ],
        )

    def test_speaker_diarization_3_speakers(self):
        tensor = torch.zeros((10, 3))
        tensor[1:4, 0] = 1
        tensor[3:8, 1] = 1
        tensor[8:10, 2] = 1

        with self.assertRaises(ValueError):
            outputs = speaker_diarization_normalize(
                tensor, 16000, ["SPEAKER_0", "SPEAKER_1"]
            )
        outputs = speaker_diarization_normalize(
            tensor, 16000, ["SPEAKER_0", "SPEAKER_1", "SPEAKER_2"]
        )
        self.assertEqual(
            outputs,
            [
                {"class": "SPEAKER_0", "start": 1 / 16000, "end": 4 / 16000},
                {"class": "SPEAKER_1", "start": 3 / 16000, "end": 8 / 16000},
                {"class": "SPEAKER_2", "start": 8 / 16000, "end": 10 / 16000},
            ],
        )
