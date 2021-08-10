import { getModelInputSnippet } from "$lib/inferenceSnippets/inputsSnippets";
import type { PipelineType, ModelData } from "$lib/interfaces/Types";

const bodyZeroShotClassification = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
    "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
})`;

const bodyTranslation = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

const bodySummarization = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

const bodyConversational = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

const bodyTableQuestionAnswering = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

const bodyQuestionAnswering = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

const bodyTextClassification = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

const bodyTokenClassification = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

const bodyTextGeneration = (model: ModelData): string =>
	`output = query(${getModelInputSnippet(model)})`;

const bodyText2TextGeneration = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

const bodyFillMask = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

const bodySentenceSimilarity = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

const bodyFeatureExtraction = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const snippetBodies: {
	[key in keyof typeof PipelineType]?: (model: ModelData) => string;
} = {
	"zero-shot-classification": bodyZeroShotClassification,
	"translation":              bodyTranslation,
	"summarization":            bodySummarization,
	"conversational":           bodyConversational,
	"table-question-answering": bodyTableQuestionAnswering,
	"question-answering":       bodyQuestionAnswering,
	"text-classification":      bodyTextClassification,
	"token-classification":     bodyTokenClassification,
	"text-generation":          bodyTextGeneration,
	"text2text-generation":     bodyText2TextGeneration,
	"fill-mask":                bodyFillMask,
	"sentence-similarity":      bodySentenceSimilarity,
	"feature-extraction":       bodyFeatureExtraction,
};