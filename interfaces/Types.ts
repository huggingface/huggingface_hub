
/// In each category, order by decreasing specificity
export enum PipelineType {
	/// nlp
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
	"automatic-speech-recognition" = "automatic-speech-recognition",
	"audio-source-separation" = "audio-source-separation",
	"voice-activity-detection" = "voice-activity-detection",
}

export const ALL_PIPELINE_TYPES = Object.keys(PipelineType) as (keyof typeof PipelineType)[];
