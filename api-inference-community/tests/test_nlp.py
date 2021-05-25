import json
from unittest import TestCase

from api_inference_community.validation import normalize_payload_nlp
from parameterized import parameterized
from pydantic.error_wrappers import ValidationError


class ValidationTestCase(TestCase):
    def test_malformed_input(self):
        bpayload = b"\xc3\x28"
        with self.assertRaises(UnicodeDecodeError):
            normalize_payload_nlp(bpayload, "tag")

    def test_accept_raw_string_for_backward_compatibility(self):
        query = "funny cats"
        bpayload = query.encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(bpayload, "tag")
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, query)


class QuestionAnsweringValidationTestCase(TestCase):
    def test_question_answering(self):
        inputs = {"question": "question", "context": "context"}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "question-answering"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(inputs, normalized_inputs)

    def test_question_answering_missing_input(self):
        inputs = {"question": "question"}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "question-answering")


class SentenceSimilarityValidationTestCase(TestCase):
    def test_sentence_similarity(self):
        source_sentence = "why is the sky blue?"
        sentences = ["this is", "a list of sentences"]
        inputs = {"source_sentence": source_sentence, "sentences": sentences}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "sentence-similarity"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(inputs, normalized_inputs)

    def test_sentence_similarity_missing_input(self):
        source_sentence = "why is the sky blue?"
        inputs = {"source_sentence": source_sentence}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "sentence-similarity")


class ZeroShotValidationTestCase(TestCase):
    def test_zero_shot_single_label(self):
        params = {"candidate_labels": "happy"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "zero-shot-classification"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_zero_shot_list_labels(self):
        params = {"candidate_labels": ["happy", "sad"]}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "zero-shot-classification"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_zero_shot_empty_list(self):
        params = {"candidate_labels": []}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "zero-shot-classification")

    def test_zero_shot_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "zero-shot-classification")

    def test_zero_shot_multi_class(self):
        params = {"candidate_labels": "happy", "multi_class": True}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "zero-shot-classification"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_zero_shot_multi_class_wrong_type(self):
        params = {"candidate_labels": "happy", "multi_class": "wrong type"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "zero-shot-classification")


class FillMaskValidationTestCase(TestCase):
    def test_fill_mask_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "fill-mask"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_fill_mask_with_top_k(self):
        params = {"top_k": 10}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "fill-mask"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_fill_mask_with_top_k_invalid_value(self):
        params = {"top_k": 0}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "fill-mask")

    def test_fill_mask_top_k_wrong_type(self):
        params = {"top_k": "wrong type"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "fill-mask")


def make_text_generation_test_case(tag):
    def valid_params():
        return [
            ("max_length", 10),
            ("top_k", 5),
            ("top_p", 0.5),
            ("repetition_penalty", 50.0),
            ("temperature", 10.0),
        ]

    def invalid_params():
        return [
            ("max_length", 1000),
            ("max_length", 0),
            ("max_length", "invalid"),
            ("top_k", 0),
            ("top_k", "invalid"),
            ("top_p", -0.1),
            ("top_p", 2.1),
            ("top_p", "invalid"),
            ("repetition_penalty", -0.1),
            ("repetition_penalty", 200.1),
            ("repetition_penalty", "invalid"),
            ("temperature", -0.1),
            ("temperature", 200.1),
            ("temperature", "invalid"),
        ]

    class TextGenerationTestCase(TestCase):
        def test_text_generation_no_params(self):
            bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
            normalized_inputs, processed_params = normalize_payload_nlp(bpayload, tag)
            self.assertEqual(processed_params, {})
            self.assertEqual(normalized_inputs, "whatever")

        @parameterized.expand(valid_params())
        def test_text_generation_valid_params(self, param_name, param_value):
            params = {param_name: param_value}
            bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
                "utf-8"
            )
            normalized_inputs, processed_params = normalize_payload_nlp(bpayload, tag)
            self.assertEqual(processed_params, params)
            self.assertEqual(normalized_inputs, "whatever")

        @parameterized.expand(invalid_params())
        def test_text_generation_invalid_params(self, param_name, param_value):
            params = {param_name: param_value}
            bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
                "utf-8"
            )
            with self.assertRaises(ValidationError):
                normalize_payload_nlp(bpayload, tag)

    return TextGenerationTestCase


class TextGenerationTestCase(make_text_generation_test_case("text-generation")):
    pass


class ConversationalTestCase(make_text_generation_test_case("conversational")):
    pass
