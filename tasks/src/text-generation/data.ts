import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A large multilingual dataset of web text. Used to pretrain generative models in various languages.",
			id:          "mc4",
		},
		{
			description: "Diverse, open source data consisting in 22 smaller, combined high-quality datasets. It was used to train GPT-Neo.",
			id:          "the_pile",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Input",
				content:
						"Once upon a time,",
				type: "text",
			},
			
		],
		outputs: [
			{
				label:   "Output",
				content:
						"Once upon a time, we knew that our ancestors were on the verge of extinction. The great explorers and poets of the Old World, from Alexander the Great to Chaucer, are dead and gone. A good many of our ancient explorers and poets have",
				type: "text",
			},
		],
	},
	id:        "text-generation",
	label:     PipelineType["text-generation"],
	libraries: TASKS_MODEL_LIBRARIES["text-generation"],
	metrics:   [
		{
			description: "Cross Entropy is a metric that calculates the difference between two probability distributions. Each probability distributions is the distribution of predicted words.",
			id:          "Cross Entropy",
		},
		{
			description: "Perplexity is the exponential of the cross-entropy loss. It evaluates the probabilities assigned to the next word by the model. Lower perplexity indicates good performance.",
			id:          "Perplexity",
		},
	],
	models: [
		{
			description: "The model from OpenAI that helped usher in the Transformer revolution.",
			id:          "gpt2",
		},
		{
			description: "A special Transformer model that can generate high-quality text for various tasks.",
			id:          "bigscience/T0pp",
		},
	],
	summary:      "Text generation is the task of...well generating text! These models can complete incomplete text or asked to perform a task such as paraphrasing. This depends on the model’s training objective (but don’t worry, we will cover both of them!).  ",
	widgetModels: ["gpt2"],
	youtubeId:    "Vpjb1lu0MDk",
};

export default taskData;
