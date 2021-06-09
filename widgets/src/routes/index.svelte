<script>
	import InferenceWidget from "$lib/InferenceWidget/InferenceWidget.svelte";
	import type { ModelData } from "../../../interfaces/Types";

	const models: ModelData[] = [
		{
			modelId: "distilbert-base-uncased",
			private: false,
			autoArchitecture: "AutoModelForMaskedLM",
			pipeline_tag: "fill-mask",
			mask_token: "[MASK]",
			tags: [],
			widgetData: [{ text: "The goal of life is [MASK]." }],
		},
		{
			modelId: "dbmdz/bert-large-cased-finetuned-conll03-english",
			private: false,
			autoArchitecture: "AutoModelForTokenClassification",
			pipeline_tag: "token-classification",
			tags: [],
			widgetData: [
				{
					text: "My name is Clara and I live in Berkeley, California. I work at this cool company called Hugging Face.",
				},
			],
		},
	];
	const model = models[0];
</script>

<h1>Welcome to SvelteKit</h1>
<p>
	Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation
</p>

{#each models as model}
	<InferenceWidget {model} apiToken="" />
{/each}
