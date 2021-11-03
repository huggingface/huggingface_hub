import os
from unittest import TestCase

from api_inference_community.validation import normalize_payload_video
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV


class ValidationTestCase(TestCase):
    def test_original_videofile(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", "archery1.mp4")
        with open(filename, "rb") as f:
            bpayload = f.read()

        payload, params = normalize_payload_video(bpayload)
        self.assertEqual(params, {})
        self.assertTrue(isinstance(payload, EncodedVideoPyAV))

        clip = payload.get_clip(0, 1)
        video = clip["video"]
        audio = clip["audio"]
        self.assertEqual(video.shape, (3, 30, 240, 320))
        self.assertEqual(audio.shape, (45056,))

    def test_secondary_file(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", "archery2.webm")
        with open(filename, "rb") as f:
            bpayload = f.read()

        payload, params = normalize_payload_video(bpayload)
        self.assertEqual(params, {})
        self.assertTrue(isinstance(payload, EncodedVideoPyAV))

        clip = payload.get_clip(1, 2)
        video = clip["video"]
        audio = clip["audio"]
        self.assertEqual(video.shape, (3, 26, 160, 284))
        self.assertEqual(audio.shape, (48128,))
