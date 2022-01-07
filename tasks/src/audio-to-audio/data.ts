import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
	],
	demo: {
		inputs: [
			{
				filename: "input.wav",
				type:     "audio",
			},
		],
		outputs: [
			{
				filename: "label-0.wav",
				type:     "audio",
			},
			{
				filename: "label-1.wav",
				type:     "audio",
			},
		],
	},
	id:        "audio-to-audio",
	label:     PipelineType["audio-to-audio"],
	libraries: TASKS_MODEL_LIBRARIES["audio-to-audio"],
	metrics:   [
		{
			description: "The Signal-to-Noise metric is the ratio of the target signal level to the background noise level. It is calculated as the logarithm of the target signal divided by the background noise, in decibels",
			id:          "snri",
		},
		{
			description: "The Signal-to-Distortion metric is the ratio of the target signal to the sum of noise, interference, and artifact errors",
			id:          "sdri",
		},
	],
	models: [
		{
			description: "A solid model of audio source separation",
			id:          "speechbrain/sepformer-wham",
		},
		{
			description: "A speech enhancement model",
			id:          "speechbrain/metricgan-plus-voicebank",
		},
	],
	summary:      "Audio-to-Audio is a family of tasks in which the input is an audio and the output is one or more generated audios. Some example tasks are speech enhancement and source separation.",
	widgetModels: ["speechbrain/sepformer-wham"],
	youtubeId:    "",
};

export default taskData;
