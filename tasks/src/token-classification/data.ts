import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A widely used dataset used to benchmark named entity recognition models.",
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
			description: "A good performing model to identify persons, locations, organizations and names of miscellaneous entities.",
			id:          "dbmdz/bert-large-cased-finetuned-conll03-english",
		},
		{
			description: "Flair models are typically the state of the art in named entity recognition tasks.",
			id:          "flair/ner-english",
		},
	],
	summary:      "Token classification is a natural language understanding task. Token classification is the task of assigning a label to each token in a sentence. Most popular token classification tasks are named entity recognition and part-of-speech tagging.",
	widgetModels: ["stanfordnlp/stanza-en"],
	youtubeId:    "wVHdVlPScxA",
};

export default taskData;
