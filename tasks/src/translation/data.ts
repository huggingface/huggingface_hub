import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A dataset of copyright free books translated into 16 different languages .",
			id:          "opus_books",
		},
		{
			description: "An example of machine translation between programming languages, this dataset consists of functions in Java and C#.",
			id:          "code_x_glue_cc_code_to_code_trans",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Input",
				content:
						"My name is Omar and I live in Zürich.",
				type: "text",
			},
			
		],
		outputs: [
			{
				label:   "Output",
				content:
						"Mein Name ist Omar und ich wohne in Zürich.",
				type: "text",
			},
		],
	},
	id:        "translation",
	label:     PipelineType["translation"],
	libraries: TASKS_MODEL_LIBRARIES["translation"],
	metrics:   [
		{
			description: "Translation is evaluated on BLEU Score. BLEU score is calculated by counting the number of shared single or subsequent tokens between the generated sequence and the reference. Subsequent n tokens are called “n-grams”. Unigram refers to a single token while bi-gram refers to token pairs and n-grams refer to n subsequent tokens. The score ranges from 0 to 1, in which 1 means the translation perfectly matched and 0 did not match at all. Read More",
			id:          "bleu",
		},
	],
	models: [
		{
			description: "A good performing model to identify persons, locations, organizations and names of miscellaneous entities.",
			id:          "https://huggingface.co/models?search=helsinki-nlp/opus-mt-",
		},
		{
			description: "A general-purpose Transformer that can be used to translate from English to German, French, or Romanian.",
			id:          "t5-base",
		},
	],
	summary:      "Translation (also called machine translation) is the task of translating text from one language to another. You can directly use a translation model, or if you can’t find the language pair you want to work with, you can fine-tune an existing multilingual translation model (like mBART or mT5) with your own training data with the source and target languages of your own use case.",
	widgetModels: ["t5-small"],
	youtubeId:    "",
};

export default taskData;
