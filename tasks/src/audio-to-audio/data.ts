import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
	],
	demo: {
		inputs: [
			{
				filename: "audio.wav",
				type:     "audio",
			},
		],
		outputs: [
			{
				filename: "audio1.wav",
				type:     "audio",
			},
			{
				filename: "audio2.wav",
				type:     "audio",
			},
		],
	},
	id:        "audio-to-audio",
	label:     PipelineType["audio-to-audio"],
	libraries: TASKS_MODEL_LIBRARIES["audio-to-audio"],
	metrics:   [
		{
			description: "",
			id:          "SNRI",
		},
		{
			description: "",
			id:          "SDRI",
		},
	],
	models: [
		{
			description: "A good audio source separation model.",
			id:          "speechbrain/sepformer-wham",
		},
		{
			description: "A speech enhancement model",
			id:          "speechbrain/metricgan-plus-voicebank",
		},
	],
	summary:      "Audio to audio is a family of tasks in which the input is an audio and the output is one or multiple audios. Some example tasks are speech enhancement and source separation.",
	widgetModels: ["speechbrain/sepformer-wham"],
	youtubeId:    "",
};

export default taskData;
