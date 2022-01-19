import type { TaskData } from "../Types";

import { PipelineType } from "../../../js/src/lib/interfaces/Types";
import { TASKS_MODEL_LIBRARIES } from "../const";

const taskData: TaskData = {
	datasets: [
		{
			description: "News articles in five different languages along with their summaries. Widely used for benchmarking multilingual summarization models.",
			id:          "mlsum",
		},
		{
			description: "English conversations and their summaries. Useful for benchmarking conversational agents.",
			id:          "samsum",
		},
	],
	demo: {
		inputs: [
			{
				label:   "Input",
				content:
						"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
				type: "text",
			},
			
		],
		outputs: [
			{
				label:   "Output",
				content:
						"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. It was the first structure to reach a height of 300 metres.",
				type: "text",
			},
		],
	},
	id:        "summarization",
	label:     PipelineType["summarization"],
	libraries: TASKS_MODEL_LIBRARIES["summarization"],
	metrics:   [
		{
			description: "The generated sequence is compared against its summary, and the overlap of tokens are counted. ROUGE-N refers to overlap of N subsequent tokens, ROUGE-1 refers to overlap of single tokens and ROUGE-2 is the overlap of two subsequent tokens.",
			id:          "rouge",
		},
	],
	models: [
		{
			description: "A strong summarization model trained on English news articles. Excels at generating factual summaries.",
			id:          "facebook/bart-large-cnn",
		},
		{
			description: "A summarization model trained on medical articles.",
			id:          "google/bigbird-pegasus-large-pubmed",
		},
	],
	summary:      "Summarization is the task of producing a shorter version of a document while preserving its important information. Some models can extract text from the original input, while other models can generate entirely new text.",
	widgetModels: ["sshleifer/distilbart-cnn-12-6"],
	youtubeId:    "yHnr5Dk2zCI",
};

export default taskData;
