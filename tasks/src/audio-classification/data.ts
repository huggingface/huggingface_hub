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
		outputs: 
		[
			{
				data: [
					{
						"label": "Up",
						"score": 0.2
					},
					{
						"label": "Down",
						"score": 0.8
					},
				],
				"type": "chart",
			},
		],
	},
	id:        "audio-classification",
	label:     PipelineType["audio-classification"],
	libraries: TASKS_MODEL_LIBRARIES["audio-classification"],
	metrics:   [
		{
			description: "The Accuracy metric is the ratio of correct predictions to the total number of cases processed. It can be calculated as: Accuracy = (TP + TN) / (TP + TN + FP + FN). Where TP is True Positive; TN is True Negative; FP is False Positive; and FN is False Negative.",
			id: "accuracy",
		},
		{
			description: "The Recall metric is the fraction of the total amount of relevant examples that were actually retrieved. It can be calculated as: Recall = TP / (TP + FN). Where TP is True Positive; and FN is False Negative.",
			id: "recall",

		},
		{
			description: "The Precision metric is the fraction of true examples among the predicted examples. It can be calculated as: Precision = TP / (TP + FP). Where TP is True Positive; and FP is False Positive.",
			id: "precision",
		},
		{
			description: "The F1 metric is the harmonic mean of the precision and recall. It can be calculated as: F1 = 2 * (precision * recall) / (precision + recall).",
			id: "f1",
		},
	],
	models: [
		{
			description: "An easy-to-use model for Command Recognition",
			id:          "speechbrain/google_speech_command_xvector",
		},
		{
			description: "A model of Emotion Recognition",
			id:          "superb/hubert-large-superb-er",
		},
	],
	summary:      "Audio classification is the task of assigning a label or class to a given audio. It can be used to recognize what command a user is giving, the emotion of a statement, or to identify a speaker.",
	widgetModels: ["speechbrain/google_speech_command_xvector"],
	youtubeId:    "",
};

export default taskData;
