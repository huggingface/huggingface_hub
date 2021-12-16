import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";
import { answers } from "../../../../views/components/Question/stores";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "A famous question answering dataset based on English articles from Wikipedia.",
			id:          "squad_v2",
		},
		{
			// TODO write proper description
			description: "A dataset of real anonymized, aggregated queries issued to the Google search engine.",
			id:          "natural_questions",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Question",
				content: "Which name is also used to describe the Amazon rainforest in English?",
				type: "text",
			},
			{
				label:   "Context",
				content: "The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle",
				type: "text",
			},
		],
		outputs: [
			{
				label:   "Answer",
				content: "Amazonia",
				type:    "text",
			},
		],
	},
	id:        "question-answering",
	label:     PipelineType["question-answering"],
	libraries: TASKS_MODEL_LIBRARIES["question-answering"],
	metrics:   [
		{
			description: "Exact Match is a metric that is based on the strict character match of the predicted answer and the ground truth, for true answers, EM is 1, even if one character is different, EM will be 0.",
			id:          "exact-match",
		},
		{
			description: " F1-Score is a useful metric if we value both false positives and false negatives equally. F1-Score is calculated over each word in the predicted sequence against the answer in the ground truth.",
			id:          "f1",
		},
	],
	models: [
		{
			description: "A strong baseline model for most question answering domains.",
			id:          "deepset/roberta-base-squad2",
		},
		{
			description: "A special model that can answer questions from tables!",
			id:          "google/tapas-base-finetuned-wtq",
		},
	],
	summary:      "Question Answering models are able to retrieve the answer of question from a given text, which is useful to search for an answer in a document. Some question answering models can event generate answers without a context!",
	widgetModels: ["deepset/roberta-base-squad2"],
	youtubeId:    "ajPx5LwJD-I",
};


export default taskData;
