import type { PipelineType, ModelData } from "../interfaces/Types";
import { getModelInputSnippet } from "./inputs";

export const bodyBasic = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}}`;

export const bodyZeroShotClassification = (model: ModelData): string =>
	`{"inputs": ${getModelInputSnippet(model)}, "parameters": {"candidate_labels": ["refund", "legal", "faq"]}}`;

export const jsSnippetBodies:
	Partial<Record<keyof typeof PipelineType, (model: ModelData) => string>> =
{
	"zero-shot-classification": bodyZeroShotClassification,
	"translation":              bodyBasic,
	"summarization":            bodyBasic,
	"conversational":           bodyBasic,
	"table-question-answering": bodyBasic,
	"question-answering":       bodyBasic,
	"text-classification":      bodyBasic,
	"token-classification":     bodyBasic,
	"text-generation":          bodyBasic,
	"text2text-generation":     bodyBasic,
	"fill-mask":                bodyBasic,
	"sentence-similarity":      bodyBasic,
	"feature-extraction":       bodyBasic,
};

export function getJsInferenceSnippet(model: ModelData, accessToken: string): string {
	const body = model.pipeline_tag && model.pipeline_tag in jsSnippetBodies
		? jsSnippetBodies[model.pipeline_tag]?.(model) ?? ""
		: "";
	
	return `async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/${model.id}",
		{
			headers: { Authorization: "Bearer ${accessToken || `{API_TOKEN}`}" },
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

export function hasJsInferenceSnippet(model: ModelData): boolean {
	return !!model.pipeline_tag && model.pipeline_tag in jsSnippetBodies;
}
