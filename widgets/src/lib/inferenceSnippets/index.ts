import { getModelInputSnippet } from "./inputs";
import type { PipelineType, ModelData } from "../interfaces/Types";

export const bodyZeroShotClassification = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
    "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
})`;

export const bodyTranslation = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const bodySummarization = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const bodyConversational = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const bodyTableQuestionAnswering = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const bodyQuestionAnswering = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const bodyTextClassification = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

export const bodyTokenClassification = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

export const bodyTextGeneration = (model: ModelData): string =>
	`output = query(${getModelInputSnippet(model)})`;

export const bodyText2TextGeneration = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

export const bodyFillMask = (model: ModelData): string =>
	`output = query({"inputs": ${getModelInputSnippet(model)}})`;

export const bodySentenceSimilarity = (model: ModelData): string =>
	`output = query({
    "inputs": ${getModelInputSnippet(model)},
})`;

export const bodyFeatureExtraction = (model: ModelData): string =>
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