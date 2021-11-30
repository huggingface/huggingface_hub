import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			// TODO write proper description
			description: "Benchmark dataset used for the task",
			id:          "squad",
		},
		{
			// TODO write proper description
			description: "Benchmark dataset used for the task",
			id:          "adversarial_qa",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Question",
				content:
						"Which name is also used to describe the Amazon rainforest in English?",
				type: "text",
			},
			{
				label:   "Context",
				content:
						"The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle",
				type: "text",
			},
		],
		outputs: [
			{
				label:   "Answer",
				content: "Amazonia",
				type: "text",
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
			description: "",
			id:          "f1",
		},
	],
	models: [
		{
			description: "Extractive QA model based on roberta",
			id:          "deepset/roberta-base-squad2",
		},
		{
			description: "DistilBERT-base-cased fine-tuned on SQUAD v1.1",
			id:          "distilbert-base-cased-distilled-squad",
		},
		{
			description: "BERT with whole word masking fine-tuned on SQuAD",
			id:          "bert-large-uncased-whole-word-masking-finetuned-squad",
		},
	],
	summary:      "Question answering is a natural language understanding task. Question answering models allow users to search for an answer in a document. Question answering models take a context and a question and output the answer.",
	widgetModels: ["deepset/roberta-base-squad2"],
};

export default taskData;
