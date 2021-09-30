import type { PipelineType, ModelData } from "../interfaces/Types";

const inputsZeroShotClassification = () =>
	`"Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"`;

const inputsTranslation = () =>
	`"inputs": "Меня зовут Вольфганг и я живу в Берлине"`;

const inputsSummarization = () =>
	`"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."`;

const inputsConversational = () =>
	`{
		"past_user_inputs": ["Which movie is the best ?"],
		"generated_responses": ["It's Die Hard for sure."],
		"text": "Can you explain why ?",
	}`;

const inputsTableQuestionAnswering = () =>
	`{
		"query": "How many stars does the transformers repository have?",
		"table": {
			"Repository": ["Transformers", "Datasets", "Tokenizers"],
			"Stars": ["36542", "4512", "3934"],
			"Contributors": ["651", "77", "34"],
			"Programming language": [
				"Python",
				"Python",
				"Rust, Python and NodeJS",
			],
		}
	}`;

const inputsQuestionAnswering = () =>
	`{
		"question": "What's my name?",
		"context": "My name is Clara and I live in Berkeley.",
	}`;

const inputsTextClassification = () =>
	`"I like you. I love you"`;

const inputsTokenClassification = () =>
	`"My name is Sarah Jessica Parker but you can call me Jessica"`;

const inputsTextGeneration = () =>
	`"Can you please let us know more details about your "`;

const inputsText2TextGeneration = () =>
	`"The answer to the universe is"`;

const inputsFillMask = (model: ModelData) =>
	`"The answer to the universe is ${model.mask_token}."`;

const inputsSentenceSimilarity = () =>
	`{
		"source_sentence": "That is a happy person",
		"sentences": [
			"That is a happy dog",
			"That is a very happy person",
			"Today is a sunny day"
		]
	}`;

const inputsFeatureExtraction = () =>
	`"Today is a sunny day and I'll get some ice cream."`;

const modelInputSnippets: {
	[key in keyof typeof PipelineType]?: (model: ModelData) => string;
} = {
	"conversational":           inputsConversational,
	"feature-extraction":       inputsFeatureExtraction,
	"fill-mask":                inputsFillMask,
	"question-answering":       inputsQuestionAnswering,
	"sentence-similarity":      inputsSentenceSimilarity,
	"summarization":            inputsSummarization,
	"table-question-answering": inputsTableQuestionAnswering,
	"text-classification":      inputsTextClassification,
	"text-generation":          inputsTextGeneration,
	"text2text-generation":     inputsText2TextGeneration,
	"token-classification":     inputsTokenClassification,
	"translation":              inputsTranslation,
	"zero-shot-classification": inputsZeroShotClassification,
};

export function getModelInputSnippet(model: ModelData): string {
	if (model.pipeline_tag) {
		const inputs = modelInputSnippets[model.pipeline_tag];
		if (inputs) {
			return inputs(model);
		}
	}
	return "No input example has been defined for this model task.";
}