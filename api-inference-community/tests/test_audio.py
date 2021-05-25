import os
from unittest import TestCase

import numpy as np
from api_inference_community.validation import ffmpeg_convert, normalize_payload_audio


class ValidationTestCase(TestCase):
    def read(self, filename: str) -> bytes:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", filename)
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    def test_original_audiofile(self):
        bpayload = self.read("sample1.flac")
        payload, params = normalize_payload_audio(bpayload, 16000)
        self.assertEqual(params, {})
        self.assertEqual(type(payload), np.ndarray)
        self.assertEqual(payload.shape, (219040,))

    def test_original_audiofile_differnt_sampling_rate(self):
        bpayload = self.read("sample1.flac")

        payload, params = normalize_payload_audio(bpayload, 48000)
        self.assertEqual(params, {})
        self.assertEqual(type(payload), np.ndarray)
        self.assertEqual(payload.shape, (3 * 219040,))

    def test_malformed_audio(self):
        bpayload = self.read("malformed.flac")
        with self.assertRaises(ValueError):
            normalize_payload_audio(bpayload, 16000)

    def test_dual_channel(self):
        bpayload = self.read("sample1_dual.ogg")
        payload, params = normalize_payload_audio(bpayload, 16000)
        self.assertEqual(payload.shape, (219520,))

    def test_original_webm(self):
        bpayload = self.read("sample1.webm")
        payload, params = normalize_payload_audio(bpayload, 16000)

    def test_ffmpeg_convert(self):
        bpayload = self.read("sample1.flac")
        sampling_rate = 16000
        self.assertEqual(len(bpayload), 282378)
        waveform, params = normalize_payload_audio(bpayload, sampling_rate)
        self.assertEqual(waveform.shape, (219040,))

        out = ffmpeg_convert(waveform, sampling_rate)
        self.assertEqual(len(out), 280204)
        waveform2, params = normalize_payload_audio(out, sampling_rate)
        self.assertEqual(waveform2.shape, (219040,))

    def test_ffmpeg_convert_8k_sampling(self):
        bpayload = self.read("sample1.flac")
        sampling_rate = 8000
        self.assertEqual(len(bpayload), 282378)
        waveform, params = normalize_payload_audio(bpayload, sampling_rate)
        self.assertEqual(waveform.shape, (109520,))

        out = ffmpeg_convert(waveform, sampling_rate)
        self.assertEqual(len(out), 258996)
        waveform2, params = normalize_payload_audio(out, sampling_rate)
        self.assertEqual(waveform2.shape, (109520,))
