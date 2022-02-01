import type { PipelineType, ModelData } from "../interfaces/Types";
import { getModelInputSnippet } from "./inputs";

export const bodyZeroShotClassification = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}, "parameters": {"candidate_labels": ["refund", "legal", "faq"]}}'`;

export const bodyTranslation = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodySummarization = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyConversational = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyTableQuestionAnswering = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyQuestionAnswering = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyTextClassification = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyTokenClassification = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyTextGeneration = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyText2TextGeneration = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodyFillMask = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}}'`;

export const bodySentenceSimilarity = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}'`;

export const bodyFeatureExtraction = (model: ModelData): string =>
	`-d '{"inputs": ${getModelInputSnippet(model, true)}'`;

export const curlSnippetBodies:
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

export function getCurlInferenceSnippet(model: ModelData, accessToken: string): string {
	const body = model.pipeline_tag && model.pipeline_tag in curlSnippetBodies
		? curlSnippetBodies[model.pipeline_tag]?.(model) ?? ""
		: "";
		
	return `curl https://api-inference.huggingface.co/models/${model.id} \\
	-X POST \\
	${body} \\
	-H "Authorization: Bearer ${accessToken}"
`;
}

export function hasCurlInferenceSnippet(model: ModelData): boolean {
	return !!model.pipeline_tag && model.pipeline_tag in curlSnippetBodies;
}
