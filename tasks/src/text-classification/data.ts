import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A widely used dataset used to benchmark multiple variants of text classification",
			id:          "glue",
		},
		{
			description: "A text classification dataset used to benchmark natural language inference models",
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
			description: "The Recall metric is the fraction of the total amount of relevant examples that were actually retrieved. It can be calculated as: Recall = TP / (TP + FN). Where TP is True Positive; and FN is False Negative",
			id:          "recall",
            
		},
		{
			description: "The Precision metric is the fraction of true examples among the predicted examples. It can be calculated as: Precision = TP / (TP + FP). Where TP is True Positive; and FP is False Positive",
			id:          "precision",
		},
		{
			description: "The F1 metric is the harmonic mean of the precision and recall. It can be calculated as: F1 = 2 * (precision * recall) / (precision + recall)",
			id: "f1",
		},
	],
	models: [
		{
			description: "A robust model trained for sentiment analysis",
			id:          "distilbert-base-uncased-finetuned-sst-2-english",
		},
		{
			description: "Multi-genre natural language inference model",
			id:          "roberta-large-mnli",
		},
	],
	summary:      "Text Classification is the task of assigning a label or class to a given text. Some use cases are sentiment analysis, natural language inference, and assessing grammatical correctness",
	widgetModels: ["distilbert-base-uncased-finetuned-sst-2-english"],
	youtubeId:    "",
};

export default taskData;
