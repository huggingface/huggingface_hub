import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A large multilingual dataset of web text. Used to pretrain GPT-like models in various languages.",
			id:          "mc4",
		},
        {
			description: "A dataset of Reddit submissions. Used to pretrain models like GPT-Neo.",
			id:          "the_pile_openwebtext2",
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
			description: "Cross Entropy is a loss metric built on entropy. It calculates the difference between two probability distributions, with probability distributions being the distributions of predicted words here.",
			id:          "Cross Entropy",
		},
        {
			description: "Perplexity is the exponential of the cross-entropy loss. Perplexity evaluates the probabilities assigned to the next word by the model, and lower perplexity indicates good performance.",
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
			id:          "bigscience/T0pp"
		}
	],
	summary:      "Text generation is the task of generating text for a given task. These models can be either asked to complete incomplete text or asked to perform a task (e.g. paraphrasing), this depends on the model’s training objective (but don’t worry, we will cover both of them). First variant of generative models is the one that is trained to predict the next word, given a sequence of words (e.g. incomplete sentence). The most popular model for this variant is GPT-2. This models are trained on data that has no labels, you just need a plain text to train your own. You can train GPT-2 to generate a wide range of documents, from code to story. Second variant of generative models is called “text-to-text” generative model. This is trained to learn mapping between a pair of text (e.g. translation from one language to another). The most popular variants of this model are T5, T0 and BART. These models are trained with multi-tasking capabilities, they can accomplish a wide range of tasks, including summarization, translation, and text classification. ",
	widgetModels: ["gpt2"],
};

export default taskData;
