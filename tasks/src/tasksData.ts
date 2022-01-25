import type { PipelineType } from "../../js/src/lib/interfaces/Types";
import type { TaskData } from "./Types";

import audioClassification from "./audio-classification/data";
import audioToAudio from "./audio-to-audio/data";
import automaticSpeechRecognition from "./automatic-speech-recognition/data";
import fillMask from "./fill-mask/data";
import imageClassification from "./image-classification/data";
import imageSegmentation from "./image-segmentation/data";
import objectDetection from "./object-detection/data";
import questionAnswering from "./question-answering/data";
import sentenceSimilarity from "./sentence-similarity/data";
import summarization from "./summarization/data";
import textToSpeech from "./text-to-speech/data";
import tokenClassification from "./token-classification/data";
import translation from "./translation/data";
import textClassification from "./text-classification/data";
import textGeneration from "./text-generation/data";


// To make comparisons easier, task order is the same as in /lib/interfaces/Types.ts
export const TASKS_DATA: Record<
	keyof typeof PipelineType,
	TaskData | undefined
> = {
	/// nlp
	"text-classification":            textClassification,
	"token-classification":           tokenClassification,
	"table-question-answering":       undefined,
	"question-answering":             questionAnswering,
	"zero-shot-classification":       undefined,
	"translation":                    translation,
	"summarization":                  summarization,
	"conversational":                 undefined,
	"feature-extraction":             undefined,
	"text-generation":                textGeneration,
	// note: we don't have a text2text-generation task, we use text-generation instead
	"text2text-generation":           undefined,
	"fill-mask":                      fillMask,
	"sentence-similarity":            sentenceSimilarity,
	/// audio
	"text-to-speech":                 textToSpeech,
	"automatic-speech-recognition":   automaticSpeechRecognition,
	"audio-to-audio":                 audioToAudio,
	"audio-classification":           audioClassification,
	"voice-activity-detection":       undefined,
	/// computer vision
	"image-classification":           imageClassification,
	"object-detection":               objectDetection,
	"image-segmentation":             imageSegmentation,
	"text-to-image":                  undefined,
	"image-to-text":                  undefined,
	/// others
	"structured-data-classification": undefined,
} as const;
