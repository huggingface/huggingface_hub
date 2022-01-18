import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A dataset of copyright-free books translated into 16 different languages.",
			id:          "opus_books",
		},
		{
			description: "An example of translation between programming languages. This dataset consists of functions in Java and C#.",
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
			description: "BLEU score is calculated by counting the number of shared single or subsequent tokens between the generated sequence and the reference. Subsequent n tokens are called “n-grams”. Unigram refers to a single token while bi-gram refers to token pairs and n-grams refer to n subsequent tokens. The score ranges from 0 to 1, where 1 means the translation perfectly matched and 0 did not match at all",
			id:          "bleu",
		},
		{
			description: "",
			id:          "sacrebleu",
		},
	],
	models: [
		{
			description: "A model that translates from English to French.",
			id:          "Helsinki-NLP/opus-mt-en-fr",
		},
		{
			description: "A general-purpose Transformer that can be used to translate from English to German, French, or Romanian.",
			id:          "t5-base",
		},
	],
	summary:      "Translation is the task of converting text from one language to another.",
	widgetModels: ["t5-small"],
	youtubeId:    "1JvfrvZgi6c",
};

export default taskData;
