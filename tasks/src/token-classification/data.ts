import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A widely used dataset useful to benchmark named entity recognition models.",
			id:          "conll2003",
		},
		{
			description: "A multilingual dataset of Wikipedia articles annotated for named entity recognition in over 150 different languages.",
			id:          "wikiann",
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
				text:   "My name is Omar and I live in Zürich.",
				tokens: [
					{
						type:  "PERSON",
						start: 11,
						end:   15,
					},
					{
						type:  "GPE",
						start: 30,
						end:   36,
					},
				],
				type: "text-with-tokens",
			},
		],
	},
	id:        "token-classification",
	label:     PipelineType["token-classification"],
	libraries: TASKS_MODEL_LIBRARIES["token-classification"],
	metrics:   [
		{
			description: "",
			id: "accuracy",
		},
		{
			description: "",
			id: "recall",

		},
		{
			description: "",
			id: "precision",
		},
		{
			description: "",
			id: "f1",
		},
	],
	models: [
		{
			description: "A robust performance model to identify people, locations, organizations and names of miscellaneous entities.",
			id:          "dslim/bert-base-NER",
		},
		{
			description: "Flair models are typically the state of the art in named entity recognition tasks.",
			id:          "flair/ner-english",
		},
	],
	summary:      "Token classification is a natural language understanding task in which a label is assigned to some tokens in a text. Some popular token classification subtasks are Named Entity Recognition (NER) and Part-of-Speech (PoS) tagging. NER models could be trained to identify specific entities in a text, such as dates, individuals and places; and PoS tagging would identify, for example, which words in a text are verbs, nouns, and punctuation marks.",
	widgetModels: ["dslim/bert-base-NER"],
	youtubeId:    "wVHdVlPScxA",
};

export default taskData;
