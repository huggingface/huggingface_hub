import type { TaskData } from "../Types";

import { PipelineType } from "../../../widgets/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "A dataset of news articles in five different languages along with their summaries. Widely used for benchmarking multilingual summarization models.",
			id:          "mlsum",
		},
        {
			description: "A dataset of English conversations and their summaries. Useful for benchmarking conversational agents.",
			id:          "samsum",
		},
	],
	demo: {
        inputs: [
			{
				label:   "Input",
				content:
						"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
				type: "text",
			},
			
		],
		outputs: [
			{
				label:   "Output",
				content:
						"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building . It was the first structure to reach a height of 300 metres . It is now taller than the Chrysler Building in New York City by 5.2 metres (17 ft) Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France .",
				type: "text",
			},
		],
	},
	id:        "summarization",
	label:     PipelineType["summarization"],
	libraries: TASKS_MODEL_LIBRARIES["summarization"],
	metrics:   [
		{
			description: "Recall-Oriented Understudy for Gisting Evaluation is the metric used for summarization. The generated sequence is compared against the summary, and the overlap of tokens are counted. ROUGE-N refers to overlap of N subsequent tokens, ROUGE-1 refers to overlap of single tokens and ROUGE-2 is the overlap of two subsequent tokens. The score is based on precision and recall. Precision is the ratio of number of overlapping words against number of total words in the generated sequence. Recall is calculated as number of overlapping words against the number of total words in the original text. For more information about ROUGE metric, check out the datasets documentation or the Hugging Face course.",
			id:          "rouge",
		},
	],
	models: [
		{
			description: "A strong summarization model trained on English news articles. Excels at generating generating factual summaries.",
			id:          "facebook/bart-large-cnn",
		},
		{
			description: "A summarization model trained on medical articles. ",
			id:          "google/bigbird-pegasus-large-pubmed",
		}
	],
	summary:      "Summarization is the task of producing a shorter version of a document while preserving the relevant and important information in the document. Summarization models have variants, being, extractive text summarization and abstractive text summarization. In extractive text summarization, we take the original text and extract important sentences instead, meanwhile, abstractive text summarization generates new text based on the original text.",
	widgetModels: ["sshleifer/distilbart-cnn-12-6"],
};

export default taskData;
