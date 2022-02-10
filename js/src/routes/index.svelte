<script>
	import InferenceWidget from "../lib/components/InferenceWidget/InferenceWidget.svelte";
	import ModeSwitcher from "../lib/components/DemoThemeSwitcher/DemoThemeSwitcher.svelte";
	import type { ModelData } from "../lib/interfaces/Types";

	const models: ModelData[] = [
		{
			id: "sgugger/resnet50d",
			pipeline_tag: "image-classification",
		},
		{
			id: "facebook/detr-resnet-50",
			pipeline_tag: "object-detection",
		},
		{
			id: "facebook/detr-resnet-50-panoptic",
			pipeline_tag: "image-segmentation",
		},
		{
			id: "julien-c/distilbert-feature-extraction",
			pipeline_tag: "feature-extraction",
			widgetData: [{ text: "Hello world" }],
		},
		{
			id: "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
			pipeline_tag: "feature-extraction",
			widgetData: [{ text: "Hello, world" }],
		},
		{
			id: "roberta-large-mnli",
			pipeline_tag: "text-classification",
			widgetData: [{ text: "I like you. </s></s> I love you." }],
		},
		{
			id: "dbmdz/bert-large-cased-finetuned-conll03-english",
			pipeline_tag: "token-classification",
			widgetData: [
				{ text: "My name is Wolfgang and I live in Berlin" },
				{ text: "My name is Sarah and I live in London" },
				{ text: "My name is Clara and I live in Berkeley, California." },
			],
		},
		{
			id: "distilbert-base-uncased-distilled-squad",
			pipeline_tag: "question-answering",
			widgetData: [
				{
					text: "Which name is also used to describe the Amazon rainforest in English?",
					context: `The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.`,
				},
			],
		},
		{
			id: "t5-base",
			pipeline_tag: "translation",
			widgetData: [{ text: "My name is Wolfgang and I live in Berlin" }],
		},
		{
			id: "facebook/bart-large-cnn",
			pipeline_tag: "summarization",
			widgetData: [
				{
					text: "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
				},
			],
		},
		{
			id: "gpt2",
			pipeline_tag: "text-generation",
			widgetData: [
				{ text: "My name is Julien and I like to" },
				{ text: "My name is Thomas and my main" },
				{ text: "My name is Mariama, my favorite" },
				{ text: "My name is Clara and I am" },
				{ text: "Once upon a time," },
			],
		},
		{
			id: "distilroberta-base",
			pipeline_tag: "fill-mask",
			mask_token: "<mask>",
			widgetData: [
				{ text: "Paris is the <mask> of France." },
				{ text: "The goal of life is <mask>." },
			],
		},
		{
			id: "facebook/bart-large-mnli",
			pipeline_tag: "zero-shot-classification",
			widgetData: [
				{
					text: "I have a problem with my iphone that needs to be resolved asap!!",
					candidate_labels: "urgent, not urgent, phone, tablet, computer",
					multi_class: true,
				},
			],
		},
		{
			id: "google/tapas-base-finetuned-wtq",
			pipeline_tag: "table-question-answering",
			widgetData: [
				{
					text: "How many stars does the transformers repository have?",
					table: {
						Repository: ["Transformers", "Datasets", "Tokenizers"],
						Stars: [36542, 4512, 3934],
						Contributors: [651, 77, 34],
						"Programming language": [
							"Python",
							"Python",
							"Rust, Python and NodeJS",
						],
					},
				},
			],
		},
		{
			id: "julien-c/wine-quality",
			pipeline_tag: "structured-data-classification",
			widgetData: [
				{
					structuredData: {
						fixed_acidity: [7.4, 7.8, 10.3],
						volatile_acidity: [0.7, 0.88, 0.32],
						citric_acid: [0.0, 0.0, 0.45],
						residual_sugar: [1.9, 2.6, 6.4],
						chlorides: [0.076, 0.098, 0.073],
						free_sulfur_dioxide: [11.0, 25.0, 5.0],
						total_sulfur_dioxide: [34.0, 67.0, 13.0],
						density: [0.9978, 0.9968, 0.9976],
						pH: [3.51, 3.2, 3.23],
						sulphates: [0.56, 0.68, 0.82],
						alcohol: [9.4, 9.8, 12.6],
					},
				},
			],
		},
		{
			id: "bigscience/T0pp",
			pipeline_tag: "text2text-generation",
		},
		{
			id: "facebook/blenderbot-400M-distill",
			pipeline_tag: "conversational",
			widgetData: [{ text: "Hey my name is Julien! How are you?" }],
		},
		{
			id: "osanseviero/BigGAN-deep-128",
			pipeline_tag: "text-to-image",
		},
		{
			id: "julien-c/kan-bayashi_csmsc_tacotron2",
			pipeline_tag: "text-to-speech",
			widgetData: [{ text: "请您说得慢些好吗" }],
		},
		{
			id: "superb/wav2vec2-base-superb-sid",
			pipeline_tag: "audio-classification",
			widgetData: [
				{
					label: "Librispeech sample 1",
					src: "https://cdn-media.huggingface.co/speech_samples/sample1.flac",
				},
			],
		},
		{
			id: "julien-c/mini_an4_asr_train_raw_bpe_valid",
			pipeline_tag: "automatic-speech-recognition",
		},
		{
			id: "facebook/wav2vec2-base-960h",
			pipeline_tag: "automatic-speech-recognition",
			widgetData: [
				{
					label: "Librispeech sample 1",
					src: "https://cdn-media.huggingface.co/speech_samples/sample1.flac",
				},
			],
		},
		{
			id: "osanseviero/full-sentence-distillroberta2",
			pipeline_tag: "sentence-similarity",
			widgetData: [
				{
					source_sentence: "That is a happy person",
					sentences: [
						"That is a happy dog",
						"That is a very happy person",
						"Today is a sunny day",
					],
				},
			],
		},
		{
			id: "speechbrain/mtl-mimic-voicebank",
			private: false,
			pipeline_tag: "audio-to-audio",
			tags: ["speech-enhancement"],
			widgetData: [],
		},
		{
			id: "speechbrain/sepformer-wham",
			private: false,
			pipeline_tag: "audio-to-audio",
			tags: ["audio-source-separation"],
			widgetData: [],
		},
		{
			id: "julien-c/DPRNNTasNet-ks16_WHAM_sepclean",
			private: false,
			pipeline_tag: "audio-to-audio",
			tags: ["audio-source-separation"],
			widgetData: [],
		},
	];
</script>

<div class="py-24">
	<ModeSwitcher />

	<div
		class="mx-4 space-y-4 lg:space-y-0 lg:grid lg:grid-cols-2 lg:gap-4 xl:grid-cols-3"
	>
		{#each models as model}
			<div>
				<a class="text-xs block mb-3 text-gray-300" href="/{model.id}">
					<code>{model.id}</code>
				</a>
				<div class="p-5 shadow-sm rounded-xl bg-white max-w-md">
					<InferenceWidget {model} />
				</div>
			</div>
		{/each}
	</div>
</div>
