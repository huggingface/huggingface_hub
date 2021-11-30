import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A nice English dataset with 1000 hours of data",
			id:          "librispeech_asr",
		},
		{
			description: "Good dataset in 60 languages with demographic information",
			id:          "common_voice",
		},
	],
	demo: {
		inputs: [
			{
				filename: "audio.wav",
				type: "audio"
			}
		],
		outputs: [
			{
				label:   "Transcript",
				content: "Today is a nice day...",
				type: "text",
			},
		],
	},
	id:        "automatic-speech-recognition",
	label:     PipelineType["automatic-speech-recognition"],
	libraries: TASKS_MODEL_LIBRARIES["automatic-speech-recognition"],
	metrics:   [
		{
			description: "",
			id:          "wer",
		},
		{
			description: "",
			id:          "cer",
		},
	],
	models: [
		{
			description: "A good generic ASR model.",
			id:          "facebook/wav2vec2-base-960h",
		},
		{
			description: "An end-to-end model that performs ASR and Speech Translation.",
			id:          "facebook/s2t-small-mustc-en-fr-st",
		}
	],
	summary:      "Automatic Speech Recognition (ASR), also known as Speech to Text (STT), is the task of transcribing a given audio into text. It has many applications such as voice user interfaces.",
	widgetModels: ["facebook/wav2vec2-base-960h"],
};

export default taskData;
