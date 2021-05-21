
/// Order by decreasing specificity
export enum PipelineType {
	"text-classification" = "text-classification",
	"token-classification" = "token-classification",
	"question-answering" = "question-answering",
	"table-question-answering" = "table-question-answering",
	"zero-shot-classification" = "zero-shot-classification",
	"translation" = "translation",
	"summarization" = "summarization",
	"text-generation" = "text-generation",
	"text2text-generation" = "text2text-generation",
	"fill-mask" = "fill-mask",
}

export const ALL_PIPELINE_TYPES = Object.keys(PipelineType) as (keyof typeof PipelineType)[];
