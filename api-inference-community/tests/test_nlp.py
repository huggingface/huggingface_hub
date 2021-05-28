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
    def test_valid_input(self):
        inputs = {"question": "question", "context": "context"}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "question-answering"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(inputs, normalized_inputs)

    def test_missing_input(self):
        inputs = {"question": "question"}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "question-answering")


class SentenceSimilarityValidationTestCase(TestCase):
    def test_valid_input(self):
        source_sentence = "why is the sky blue?"
        sentences = ["this is", "a list of sentences"]
        inputs = {"source_sentence": source_sentence, "sentences": sentences}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "sentence-similarity"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(inputs, normalized_inputs)

    def test_missing_input(self):
        source_sentence = "why is the sky blue?"
        inputs = {"source_sentence": source_sentence}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "sentence-similarity")


class ConversationalValidationTestCase(TestCase):
    def test_valid_inputs(self):
        past_user_inputs = ["Which movie is the best ?"]
        generated_responses = ["It's Die Hard for sure."]
        text = "Can you explain why ?"

        inputs = {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": text,
        }

        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "conversational"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(inputs, normalized_inputs)


class TableQuestionAnsweringValidationTestCase(TestCase):
    def test_valid_input(self):
        query = "How many stars does the transformers repository have?"
        table = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
        }

        inputs = {"query": query, "table": table}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "table-question-answering"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(inputs, normalized_inputs)

    def test_invalid_question(self):
        query = "How many stars does the transformers repository have?"
        table = "Invalid table"
        inputs = {"query": query, "table": table}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "table-question-answering")

    def test_invalid_query(self):
        query = {"not a": "query"}
        table = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
        }
        inputs = {"query": query, "table": table}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "table-question-answering")

    def test_no_table(self):
        query = "How many stars does the transformers repository have?"
        inputs = {
            "query": query,
        }
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "table-question-answering")

    def test_no_query(self):
        table = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
        }
        inputs = {"table": table}
        bpayload = json.dumps({"inputs": inputs}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "table-question-answering")


class SummarizationValidationTestCase(TestCase):
    def test_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "summarization"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_valid_min_length(self):
        params = {"min_length": 10}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "summarization"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_invalid_negative_min_length(self):
        params = {"min_length": -1}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )

    def test_invalid_large_min_length(self):
        params = {"min_length": 1000}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )

    def test_invalid_type_min_length(self):
        params = {"min_length": "invalid"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )

    def test_valid_max_length(self):
        params = {"max_length": 10}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "summarization"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_invalid_negative_max_length(self):
        params = {"max_length": -1}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )

    def test_invalid_large_max_length(self):
        params = {"max_length": 1000}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )

    def test_invalid_type_max_length(self):
        params = {"max_length": "invalid"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )

    def test_invalid_min_length_larger_than_max_length(self):
        params = {"min_length": 20, "max_length": 10}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalized_inputs, processed_params = normalize_payload_nlp(
                bpayload, "summarization"
            )


class ZeroShotValidationTestCase(TestCase):
    def test_single_label(self):
        params = {"candidate_labels": "happy"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "zero-shot-classification"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_list_labels(self):
        params = {"candidate_labels": ["happy", "sad"]}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "zero-shot-classification"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_empty_list(self):
        params = {"candidate_labels": []}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "zero-shot-classification")

    def test_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "zero-shot-classification")

    def test_multi_label(self):
        params = {"candidate_labels": "happy", "multi_label": True}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "zero-shot-classification"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_multi_label_wrong_type(self):
        params = {"candidate_labels": "happy", "multi_label": "wrong type"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "zero-shot-classification")


class FillMaskValidationTestCase(TestCase):
    def test_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "fill-mask"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_top_k(self):
        params = {"top_k": 10}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "fill-mask"
        )
        self.assertEqual(processed_params, params)
        self.assertEqual(normalized_inputs, "whatever")

    def test_top_k_invalid_value(self):
        params = {"top_k": 0}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "fill-mask")

    def test_top_k_wrong_type(self):
        params = {"top_k": "wrong type"}
        bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
            "utf-8"
        )
        with self.assertRaises(ValidationError):
            normalize_payload_nlp(bpayload, "fill-mask")


def make_text_generation_test_case(tag):
    def valid_params():
        return [
            ("max_new_tokens", 10),
            ("top_k", 5),
            ("top_p", 0.5),
            ("max_time", 20.0),
            ("repetition_penalty", 50.0),
            ("temperature", 10.0),
            ("return_full_text", True),
            ("num_return_sequences", 2),
        ]

    def invalid_params():
        return [
            ("min_length", 1000),
            ("min_length", 0),
            ("min_length", "invalid"),
            ("max_length", 1000),
            ("max_length", 0),
            ("max_length", "invalid"),
            ("top_k", 0),
            ("top_k", "invalid"),
            ("top_p", -0.1),
            ("top_p", 2.1),
            ("top_p", "invalid"),
            ("max_time", -0.1),
            ("max_time", 120.1),
            ("max_time", "invalid"),
            ("repetition_penalty", -0.1),
            ("repetition_penalty", 200.1),
            ("repetition_penalty", "invalid"),
            ("temperature", -0.1),
            ("temperature", 200.1),
            ("temperature", "invalid"),
            ("return_full_text", "invalid"),
            ("num_return_sequences", -1),
            ("num_return_sequences", 100),
        ]

    class TextGenerationTestCase(TestCase):
        def test_no_params(self):
            bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
            normalized_inputs, processed_params = normalize_payload_nlp(bpayload, tag)
            self.assertEqual(processed_params, {})
            self.assertEqual(normalized_inputs, "whatever")

        @parameterized.expand(valid_params())
        def test_valid_params(self, param_name, param_value):
            params = {param_name: param_value}
            bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
                "utf-8"
            )
            normalized_inputs, processed_params = normalize_payload_nlp(bpayload, tag)
            self.assertEqual(processed_params, params)
            self.assertEqual(normalized_inputs, "whatever")

        @parameterized.expand(invalid_params())
        def test_invalid_params(self, param_name, param_value):
            params = {param_name: param_value}
            bpayload = json.dumps({"inputs": "whatever", "parameters": params}).encode(
                "utf-8"
            )
            with self.assertRaises(ValidationError):
                normalize_payload_nlp(bpayload, tag)

    return TextGenerationTestCase


class Text2TextGenerationTestCase(
    make_text_generation_test_case("text2text-generation")
):
    pass


class TextGenerationTestCase(make_text_generation_test_case("text-generation")):
    pass


class TasksWithOnlyInputStringTestCase(TestCase):
    def test_feature_extraction_accept_string_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "feature-extraction"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_fill_mask_accept_string_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "fill-mask"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_text_classification_accept_string_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "text-classification"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_token_classification_accept_string_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "token-classification"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")

    def test_translation_accept_string_no_params(self):
        bpayload = json.dumps({"inputs": "whatever"}).encode("utf-8")
        normalized_inputs, processed_params = normalize_payload_nlp(
            bpayload, "translation"
        )
        self.assertEqual(processed_params, {})
        self.assertEqual(normalized_inputs, "whatever")
