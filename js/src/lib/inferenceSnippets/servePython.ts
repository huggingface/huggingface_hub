import type { PipelineType, ModelData } from "../interfaces/Types";
import { getModelInputSnippet } from "./inputs";

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

export const pythonSnippetBodies:
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

const pythonSnippetHeader = (modelId: string, accessToken: string = "") =>
	`import requests

API_URL = "https://api-inference.huggingface.co/models/${modelId}"
headers = {"Authorization": ${accessToken ? `"Bearer ${accessToken}"` : `f"Bearer {API_TOKEN}"`}}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

`;

export function getPythonInferenceSnippet(model: ModelData, accessToken: string): string {
	const header = pythonSnippetHeader(model.id, accessToken);
	const body = model.pipeline_tag && model.pipeline_tag in pythonSnippetBodies
		? pythonSnippetBodies[model.pipeline_tag]?.(model) ?? ""
		: "";
	return header + body;
}

export function hasPythonInferenceSnippet(model: ModelData): boolean {
	return !!model.pipeline_tag && model.pipeline_tag in pythonSnippetBodies;
}
