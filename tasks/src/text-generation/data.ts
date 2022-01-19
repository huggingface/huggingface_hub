import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A large multilingual dataset of text crawled from the web.",
			id:          "mc4",
		},
		{
			description: "Diverse open-source data consisting of 22 smaller high-quality datasets. It was used to train GPT-Neo.",
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
			description: "Cross Entropy is a metric that calculates the difference between two probability distributions. Each probability distribution is the distribution of predicted words",
			id:          "Cross Entropy",
		},
		{
			description: "The Perplexity metric is the exponential of the cross-entropy loss. It evaluates the probabilities assigned to the next word by the model. Lower perplexity indicates better performance",
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
	summary:      "Generating text is the task of producing new text. These models can, for example, fill in incomplete text or paraphrase.",
	widgetModels: ["gpt2"],
	youtubeId:    "Vpjb1lu0MDk",
};

export default taskData;
