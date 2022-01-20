import { PipelineType } from "../../js/src/lib/interfaces/Types";
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

export const TASKS_DATA: Partial<Record<keyof typeof PipelineType, TaskData>> = {
	"audio-classification":         audioClassification,
	"audio-to-audio":               audioToAudio,
	"automatic-speech-recognition": automaticSpeechRecognition,
	"fill-mask":                    fillMask,
	"image-classification":         imageClassification,
	"image-segmentation":           imageSegmentation,
	"object-detection":             objectDetection,
	"question-answering":           questionAnswering,
	"sentence-similarity":          sentenceSimilarity,
	"summarization":                summarization,
	"text-classification":          textClassification,
	"text-generation":              textGeneration,
	"text-to-speech":               textToSpeech,
	"token-classification":         tokenClassification,
	"translation":                  translation,
} as const;
