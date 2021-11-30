import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "Thousands of short audio clips of a single speaker.",
			id:          "LJ Speech Dataset",
		},
		{
			description: "Multi-speaker English dataset.",
			id:          "LibriTTS",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Input",
				content:
						"Which name is also used to describe the Amazon rainforest in English?",
				type: "text",
			},
			
		],
		outputs: [
			{
				filename: "audio.wav",
				type: "audio"
			}
		],
	},
	id:        "text-to-speech",
	label:     PipelineType["text-to-speech"],
	libraries: TASKS_MODEL_LIBRARIES["text-to-speech"],
	metrics:   [
	],
	models: [
		{
			description: "",
			id:          "espnet/kan-bayashi_ljspeech_vits",
		}
	],
	summary:      "Text to Speech (TTS), is the task of generating natural sounding speech given a text input. Text to Speech can be extended to having a single model that generates speech for multiple speakers and multiple languages.",
	widgetModels: ["espnet/kan-bayashi_ljspeech_vits"],
};

export default taskData;
