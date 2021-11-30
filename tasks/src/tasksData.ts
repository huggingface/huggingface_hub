import { PipelineType } from "../../widgets/src/lib/interfaces/Types";
import type { TaskData } from "./Types";
import audioClassification from "./audio-classification/data";
import audioToAudio from "./audio-to-audio/data";
import automaticSpeechRecognition from "./text-to-speech/data";
import objectDetection from "./object-detection/data";
import questionAnswering from "./question-answering/data";
import textToSpeech from "./text-to-speech/data";

export const TASKS_DATA: Partial<Record<keyof typeof PipelineType, TaskData>> = {
	"audio-classification": audioClassification,
	"audio-to-audio": audioToAudio,
	"automatic-speech-recognition": automaticSpeechRecognition,
	"object-detection":   objectDetection,
	"question-answering": questionAnswering,
	"text-to-speech": textToSpeech,
} as const;
