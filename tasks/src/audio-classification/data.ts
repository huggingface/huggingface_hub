import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A benchmark of 10 different audio tasks.",
			id:          "superb",
		},
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
				label:   "Down",
				content: "0.8",
				type:    "text",
			},
			{
				label:   "Up",
				content: "0.2",
				type:    "text",
			},
		],
	},
	id:        "audio-classification",
	label:     PipelineType["audio-classification"],
	libraries: TASKS_MODEL_LIBRARIES["audio-classification"],
	metrics:   [
		{
			description: "",
			id:          "accuracy",
		},
		{
			description: "",
			id:          "f1",
		},
	],
	models: [
		{
			description: "A simple to use model for Command Recognition",
			id:          "speechbrain/google_speech_command_xvector",
		},
		{
			description: "A Emotion Recognition model",
			id:          "superb/hubert-large-superb-er",
		},
	],
	summary:      "Audio classification is the task of assigning a label to a given audio. It can be used for recognizing which command the user is giving, the emotion of an utterance or identifying a speaker!",
	widgetModels: ["speechbrain/google_speech_command_xvector"],
	youtubeId:    "",
};

export default taskData;
