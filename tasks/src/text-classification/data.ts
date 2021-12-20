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
				content:
						"I love Hugging Face!",
				type: "text",
			},
			
		],
		outputs: [
			{
				label:   "Output",
				content:
						"A JSON input of bars.",
				type: "text",
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
			description: "A good performing model on trained on sentiment analysis.",
			id:          "distilbert-base-uncased-finetuned-sst-2-english",
		},
		{
			description: "Strong Multi-Genre Natural Language Inference model.",
			id:          "roberta-large-mnli",
		},
	],
	summary:      "Text Classification is the task of classifying a text. There are various text classification tasks, such as Natural Language Inference, Sentiment Analysis and Linguistic Acceptibility.",
	widgetModels: ["distilbert-base-uncased-finetuned-sst-2-english"],
	youtubeId:    "",
};

export default taskData;
