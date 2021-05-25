import os
from unittest import TestCase

import numpy as np
from app.validation import normalize_payload_audio


class ValidationTestCase(TestCase):
    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_original_audiofile(self):
        bpayload = self.read("sample1.flac")

        payload, params = normalize_payload_audio(bpayload)
        self.assertEqual(params, {})
        self.assertEqual(type(payload), np.ndarray)
        self.assertEqual(payload.shape, (219040,))

    def test_malformed_audio(self):
        bpayload = self.read("malformed.flac")
        with self.assertRaises(ValueError):
            normalize_payload_audio(bpayload)

    def test_dual_channel(self):
        bpayload = self.read("sample1_dual.ogg")
        payload, params = normalize_payload_audio(bpayload)
        self.assertEqual(payload.shape, (219520,))

    def test_original_webm(self):
        bpayload = self.read("sample1.webm")
        payload, params = normalize_payload_audio(bpayload)
