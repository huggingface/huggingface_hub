import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A widely used dataset used to benchmark multiple variants of text classification.",
			id:          "glue",
		},
		{
			description: "A text classification dataset used to benchmark natural language inference models.",
			id:          "snli",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Input",
				content: "I love Hugging Face!",
				type: "text",
			},
			
		],
		outputs: [
			{
				"type": "chart",
				data: [
					{
						"label": "POSITIVE",
						"score": 0.90
					},
					{
						"label": "NEUTRAL",
						"score": 0.10
					},
					{
						"label": "NEGATIVE",
						"score": 0.00
					}
				]
			},
		],
	},
	id:        "text-classification",
	label:     PipelineType["text-classification"],
	libraries: TASKS_MODEL_LIBRARIES["text-classification"],
	metrics:   [
		{
			description: "",
			id:          "accuracy",
		},
		{
			description: "",
			id:          "f1",
		},
		{
			description: "",
			id:          "recall",
            
		},
		{
			description: "",
			id:          "precision",
		},
	],
	models: [
		{
			description: "A good performing model trained on sentiment analysis.",
			id:          "distilbert-base-uncased-finetuned-sst-2-english",
		},
		{
			description: "Strong multi-genre natural language inference model.",
			id:          "roberta-large-mnli",
		},
	],
	summary:      "Text Classification is the task of assigning a label or class to a given text. Some example tasks are sentiment analysis, natural language inference, and grammatical correctness.",
	widgetModels: ["distilbert-base-uncased-finetuned-sst-2-english"],
	youtubeId:    "",
};

export default taskData;
