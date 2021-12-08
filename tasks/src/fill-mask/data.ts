import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A common dataset used for pretraining models like BERT.",
			id:          "wikipedia",
		},
		{
			description: "A large English dataset of web text. Used to pretrain models like T5.",
			id:          "c4",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Input",
				content: "The goal of life is [MASK].",
				type:    "text",
			},
			
		],
		outputs: [
			{
				"type": "chart",
				data: [
					{
						"label": "life",
						"score": 0.10933306068181992
					},
					{
						"label": "survival",
						"score": 0.039418820291757584
					},
					{
						"label": "love",
						"score": 0.032930586487054825
					},
					{
						"label": "freedom",
						"score": 0.03009609691798687
					},
					{
						"label": "simplicity",
						"score": 0.02496715635061264
					}
				]
			},
		],
	},
	id:        "fill-mask",
	label:     PipelineType["fill-mask"],
	libraries: TASKS_MODEL_LIBRARIES["fill-mask"],
	metrics:   [
		{
			description: "Cross Entropy is a loss metric built on entropy. It calculates the difference between two probability distributions, with probability distributions being the distributions of predicted words here.",
			id:          "cross_entropy",
		},
		{
			description: "Perplexity is the exponential of the cross-entropy loss. Perplexity evaluates the probabilities assigned to the next word by the model, and lower perplexity indicates good performance. ",
			id:          "perplexity",
		},
	],
	models: [
		{
			description: "A smaller, faster version of BERT.",
			id:          "distilbert-base-uncased",
		},
		{
			description: "A multilingual model trained on 100 languages.",
			id:          "xlm-roberta-base",
		},
	],
	summary:      "Masked language modeling is the task of predicting which words should fill in the blanks of a sentence. The task doesn’t require labelled data, it’s trained by masking a couple of words in the sentences and model is expected to guess the masked word.",
	widgetModels: ["bert-base-uncased"],
	youtubeId:    "",
};

export default taskData;
