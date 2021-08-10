
/// Coarse-grained task classification.
///
/// This type is used to determine which inference API & widget
/// we want to display for each given model.
///
/// As such, they describe the "shape" of each model's API (inputs and outputs)
/// so the number of different types is not expected to grow very significantly over time.
///
/// In each category, order by decreasing specificity
/// The order can influence which default pipeline tag is affected to a model (if unspecified in model card)
export enum PipelineType {
	/// nlp
	"text-classification"                                     = "Text Classification",
	"token-classification"                                    = "Token Classification",
	"table-question-answering"                                = "Table Question Answering",
	"question-answering"                                      = "Question Answering",
	"zero-shot-classification"                                = "Zero-Shot Classification",
	"translation"                                             = "Translation",
	"summarization"                                           = "Summarization",
	"conversational"                                          = "Conversational",
	"feature-extraction"                                      = "Feature Extraction",
	"text-generation"                                         = "Text Generation",
	"text2text-generation"                                    = "Text2Text Generation",
	"fill-mask"                                               = "Fill-Mask",
	"sentence-similarity"                                     = "Sentence Similarity",
	/// audio
	"text-to-speech"                                          = "Text-to-Speech",
	"automatic-speech-recognition"                            = "Automatic Speech Recognition",
	"audio-to-audio"                                          = "Audio-to-Audio",
	"voice-activity-detection"                                = "Voice Activity Detection",
	/// computer vision
	"image-classification"                                    = "Image Classification",
	"object-detection"                                        = "Object Detection",
	"image-segmentation"                                      = "Image Segmentation",
	"text-to-image"                                           = "Text-to-Image",
	/// others
	"structured-data-classification"                          = "Structured Data Classification",
}


/// Finer-grained task classification
///
/// This is used in a model card's `model-index` metadata.
/// (see https://github.com/huggingface/huggingface_hub/blame/main/modelcard.md for spec)
/// and is a more granular classification that can grow significantly over time
/// as we provide support for more ML tasks.
///
/// We decide to keep it flat (non-hierchical) for simplicity and consistency.
export enum FinerGrainedTaskType {
	/// nlp
	"named-entity-recognition"                                = "Named Entity Recognition",
	"part-of-speech-tagging"                                  = "Part-Of-Speech Tagging",
	/// audio
	"audio-source-separation"                                 = "Audio Source Separation",
	"speech-enhancement"                                      = "Speech Enhancement",
}


export const ALL_PIPELINE_TYPES = Object.keys(PipelineType) as (keyof typeof PipelineType)[];

/*
 * Specification of tag icon color.
 */
export const PIPELINE_TAG_ICO_CLASS: {
	[key in keyof typeof PipelineType]?: string;
} = {
	"audio-to-audio": "tag-ico-blue",
	"automatic-speech-recognition": "tag-ico-yellow",
	"conversational": "tag-ico-green",
	"fill-mask": "tag-ico-red",
	"feature-extraction": "tag-ico-red",
	"image-classification": "tag-ico-blue",
	"image-segmentation": "tag-ico-green",
	"object-detection": "tag-ico-orange",
	"question-answering": "tag-ico-blue",
	"sentence-similarity": "tag-ico-orange",
	"structured-data-classification": "tag-ico-indigo",
	"summarization": "tag-ico-indigo",
	"table-question-answering": "tag-ico-green",
	"token-classification": "tag-ico-blue",
	"text2text-generation": "tag-ico-indigo",
	"text-classification": "tag-ico-orange",
	"text-generation": "tag-ico-indigo",
	"text-to-image": "tag-ico-orange",
	"text-to-speech": "tag-ico-yellow",
	"translation": "tag-ico-green",
	"voice-activity-detection": "tag-ico-red",
	"zero-shot-classification": "tag-ico-yellow",
};

/*
 * Specification of pipeline tag display order.
 */
export const PIPELINE_TAGS_DISPLAY_ORDER: Array<keyof typeof PipelineType> = [
	/// nlp
	"fill-mask",
	"question-answering",
	"summarization",
	"table-question-answering",
	"text-classification",
	"text-generation",
	"text2text-generation",
	"token-classification",
	"translation",
	"zero-shot-classification",
	"sentence-similarity",
	"conversational",
	"feature-extraction",
	/// audio
	"text-to-speech",
	"automatic-speech-recognition",
	"audio-to-audio",
	"voice-activity-detection",
	/// computer vision
	"image-classification",
	"object-detection",
	"image-segmentation",
	"text-to-image",
	/// others
	"structured-data-classification",
];

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
