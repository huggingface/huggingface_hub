
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
