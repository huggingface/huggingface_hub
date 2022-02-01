import type { PipelineType, ModelData } from "../interfaces/Types";
import { getModelInputSnippet } from "./inputs";

export const bodyZeroShotClassification = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}, "parameters": {"candidate_labels": ["refund", "legal", "faq"]}}`;

export const bodyTranslation = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodySummarization = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyConversational = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyTableQuestionAnswering = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyQuestionAnswering = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyTextClassification = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyTokenClassification = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyTextGeneration = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyText2TextGeneration = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyFillMask = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodySentenceSimilarity = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}`;

export const bodyFeatureExtraction = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}`;

export const nodeSnippetBodies:
	Partial<Record<keyof typeof PipelineType, (model: ModelData) => string>> =
{
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

export function getNodeInferenceSnippet(model: ModelData, accessToken: string): string {
	const body = model.pipeline_tag && model.pipeline_tag in nodeSnippetBodies
		? nodeSnippetBodies[model.pipeline_tag]?.(model) ?? ""
		: "";
	
	return `import fetch from "node-fetch";

async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/${model.id}",
		{
			headers: { Authorization: \`Bearer ${accessToken}\` },
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query(${body}).then((response) => {
	console.log(JSON.stringify(response));
});`;
}

export function hasNodeInferenceSnippet(model: ModelData): boolean {
	return !!model.pipeline_tag && model.pipeline_tag in nodeSnippetBodies;
}
