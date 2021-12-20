import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A large-scale dataset of queries from Bing search together with their relevant passages from various web sources.",
			id:          "ms_marco",
		},
		{
			description: "A dataset of question and answer pairs from the StackExchange platform.",
			id:          "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
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
			description: "The reciprocal rank is a measure used to rank the relevancy of documents given a set of documents to search in. Reciprocal Rank is the reciprocal of the rank of the document retrieved, meaning, if the rank is 3, the Reciprocal Rank is 0.33, if the rank is 1, the Reciprocal Rank is 1. The queries’ Reciprocal Ranks are averaged to calculate the MRR. Spearman’s Rank Correlation Coefficient (given in hf.co/metrics)",
			id:          "Mean Reciprocal Rank",
		},
		{
			description: "Similarity of embeddings is evaluated mainly on cosine similarity. It’s calculated as the cosine of the angle between two vectors. Cosine similarity is particularly useful when your documents do not have the same length.",
			id:          "Cosine Similarity",
		},
	],
	models: [
		{
			description: "Based on the Sentence Transformers library, this model provides the best quality embeddings to compare text pairs.",
			id:          "sentence-transformers/all-mpnet-base-v2",
		},
		{
			description: "A multilingual FAQ retrieval model, also based on Sentence Transformers.",
			id:          "clips/mfaq",
		},
	],
	summary:      "Sentence similarity is the task of determining how similar two texts are. Sentence similarity models take two text and turn them into embeddings and calculate the similarity between two embeddings, this similarity is in cosine similarity. This task is particularly useful for semantic search and clustering. Sentence similarity has various subtasks, such as passage ranking or semantic textual similarity.",
	widgetModels: ["sentence-transformers/all-MiniLM-L6-v2"],
	youtubeId:    "",
};

export default taskData;
