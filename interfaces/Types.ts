
/// Order by decreasing specificity
export enum PipelineType {
	"text-classification" = "text-classification",
	"token-classification" = "token-classification",
	"table-question-answering" = "table-question-answering",
	"question-answering" = "question-answering",
	"zero-shot-classification" = "zero-shot-classification",
	"translation" = "translation",
	"summarization" = "summarization",
	"text-generation" = "text-generation",
	"text2text-generation" = "text2text-generation",
	"fill-mask" = "fill-mask",
	/// audio
	"text-to-speech" = "text-to-speech",
}

export const ALL_PIPELINE_TYPES = Object.keys(PipelineType) as (keyof typeof PipelineType)[];
