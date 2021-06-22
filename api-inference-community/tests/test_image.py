import os
from unittest import TestCase

import PIL
from api_inference_community.validation import normalize_payload_image


class ValidationTestCase(TestCase):
    def test_original_imagefile(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", "plane.jpg")
        with open(filename, "rb") as f:
            bpayload = f.read()

        payload, params = normalize_payload_image(bpayload)
        self.assertEqual(params, {})
        self.assertTrue(isinstance(payload, PIL.Image.Image))
        self.assertEqual(payload.size, (300, 200))

    def test_secondary_file(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "samples", "plane2.jpg")
        with open(filename, "rb") as f:
            bpayload = f.read()

        payload, params = normalize_payload_image(bpayload)
        self.assertEqual(params, {})
        self.assertTrue(isinstance(payload, PIL.Image.Image))
        self.assertEqual(payload.size, (2560, 1440))
