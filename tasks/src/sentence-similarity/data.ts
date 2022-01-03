import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "Question created by crowdworkers about Wikipedia articles.",
			id:          "squad",
		},
		{
			description: "Queries from Bing with relevant passages from various web sources.",
			id:          "ms_marco",
		},
		
	],
	demo: {
		inputs: [

			
			{
				label:   "Source Sentence",
				content: "Machine learning is so easy.",
				type:    "text",
			},
			{
				label:   "Sentences to compare to",
				content: "Deep learning is so effortless.",
				type:    "text",
			},
			{
				label:   "",
				content: "This so hard, just like rocket science.",
				type:    "text",
			},
			{
				label:   "",
				content: "I can't believe how I struggled at this.",
				type:    "text",
			},
			
		],
		outputs: [
			{
				"type": "chart",
				data: [
					{
						"label": "Deep learning is so effortless.",
						"score": 0.623
					},
					{
						"label": "This used to be hard, just like rocket science.",
						"score": 0.413
					},
					{
						"label": "I can't believe how I struggled at this.",
						"score": 0.256
					},
				]
			},
		],
	},
	id:        "sentence-similarity",
	label:     PipelineType["sentence-similarity"],
	libraries: TASKS_MODEL_LIBRARIES["sentence-similarity"],
	metrics:   [
		{
			description: "The reciprocal rank is a measure used to rank the relevancy of documents given a set of documents. Reciprocal Rank is the reciprocal of the rank of the document retrieved, meaning, if the rank is 3, the Reciprocal Rank is 0.33. If the rank is 1, the Reciprocal Rank is 1.",
			id:          "Mean Reciprocal Rank",
		},
		{
			description: "The similarity of embeddings is evaluated mainly on cosine similarity. It’s calculated as the cosine of the angle between two vectors. It is particularly useful when your documents do not have the same length.",
			id:          "Cosine Similarity",
		},
	],
	models: [
		{
			description: "This is a good model that works for sentences and paragraphs and can be used for clustering and semantic search.",
			id:          "sentence-transformers/all-mpnet-base-v2",
		},
		{
			description: "A multilingual model trained for FAQ retrieval.",
			id:          "clips/mfaq",
		},
	],
	summary:      "Sentence similarity is the task of determining how similar two texts are. Sentence similarity models conver texts into embeddings and calculate the similarity between these embeddings. This task is particularly useful for semantic search and clustering. Sentence similarity has various subtasks, such as passage ranking or semantic textual similarity.",
	widgetModels: ["sentence-transformers/all-MiniLM-L6-v2"],
	youtubeId:    "",
};

export default taskData;
