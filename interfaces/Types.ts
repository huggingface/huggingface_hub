
/// In each category, order by decreasing specificity
/// The order can influence which default pipeline tag is affected to a model (if unspecified in model card)
export enum PipelineType {
	/// nlp
	"text-classification" = "text-classification",
	"token-classification" = "token-classification",
	"table-question-answering" = "table-question-answering",
	"question-answering" = "question-answering",
	"zero-shot-classification" = "zero-shot-classification",
	"translation" = "translation",
	"summarization" = "summarization",
	"conversational" = "conversational",
	"feature-extraction" = "feature-extraction",
	"text-generation" = "text-generation",
	"text2text-generation" = "text2text-generation",
	"fill-mask" = "fill-mask",
	"sentence-similarity" = "sentence-similarity",
	/// audio
	"text-to-speech" = "text-to-speech",
	"automatic-speech-recognition" = "automatic-speech-recognition",
	"audio-to-audio" = "audio-to-audio",
	"audio-source-separation" = "audio-source-separation",
	"voice-activity-detection" = "voice-activity-detection",
	/// computer vision
	"image-classification" = "image-classification",
	"object-detection" = "object-detection",
	"image-segmentation" = "image-segmentation",
	// others
	"structured-data-classification" = "structured-data-classification",
}

export const ALL_PIPELINE_TYPES = Object.keys(PipelineType) as (keyof typeof PipelineType)[];

export const PIPELINE_TYPE_PRETTY_NAMES: { [key in PipelineType]: string } = {
	/// nlp
	"text-classification":                                      "Text Classification",
	"token-classification":                                     "Token Classification",
	"table-question-answering":                                 "Table Question Answering",
	"question-answering":                                       "Question Answering",
	"zero-shot-classification":                                 "Zero-Shot Classification",
	"translation":                                              "Translation",
	"summarization":                                            "Summarization",
	"conversational":                                           "Conversational",
	"feature-extraction":                                       "Feature Extraction",
	"text-generation":                                          "Text Generation",
	"text2text-generation":                                     "Text2Text Generation",
	"fill-mask":                                                "Fill-Mask",
	"sentence-similarity":                                      "Sentence Similarity",
	/// audio
	"text-to-speech":                                           "Text-to-Speech",
	"automatic-speech-recognition":                             "Automatic Speech Recognition",
	"audio-to-audio":                                           "Audio-to-Audio",
	"audio-source-separation":                                  "Audio Source Separation",
	"voice-activity-detection":                                 "Voice Activity Detection",
	/// computer vision
	"image-classification":                                     "Image Classification",
	"object-detection":                                         "Object Detection",
	"image-segmentation":                                       "Image Segmentation",
	/// others
	"structured-data-classification":                           "Structured Data Classification",
};



/**
 * Public interface for model metadata
 */
export interface ModelData {
	/**
	 * id of model (e.g. 'user/repo_name')
	 */
	modelId: string;
	/**
	 * is this model private?
	 */
	private?: boolean;
	/**
	 * this dictionary has useful information about the model configuration
	 */
	config?: Record<string, any>;
	/**
	 * all the model tags
	 */
	tags?: string[];
	/**
	 * this is transformers-specific
	 */
	autoArchitecture?: string;
	/**
	 * Pipeline type
	 */
	pipeline_tag?: (keyof typeof PipelineType) | undefined;
	/**
	 * for relevant models, get mask token
	 */
	mask_token?: string | undefined;
	/**
	 * Example data that will be fed into the widget.
	 *
	 * can be set in the model card metadata (under `widget`),
	 * or by default in `DefaultWidget.ts`
	 */
	widgetData?: Record<string, any>[] | undefined;
}
