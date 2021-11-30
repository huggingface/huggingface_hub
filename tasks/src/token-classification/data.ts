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
				label:   "Output",
				content:
						"My name is Omar and I live in Zürich.",
				type: "text",
			},
		],
	},
	id:        "token-classification",
	label:     PipelineType["token-classification"],
	libraries: TASKS_MODEL_LIBRARIES["token-classification"],
	metrics:   [
		{
			description: "",
			id:          "Accuracy",
		},
        {
			description: "",
			id:          "F1-Score",
		},
        {
			description: "",
			id:          "Recall",
            
		},
        {
			description: "",
			id:          "Precision",
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
		}
	],
	summary:      "Token classification is a natural language understanding task. Token classification is the task of assigning a label to each token in a sentence. Most popular token classification tasks are named entity recognition and part-of-speech tagging.",
	widgetModels: ["stanfordnlp/stanza-en"],
};

export default taskData;
