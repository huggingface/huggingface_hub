<script>
	import InferenceWidget from "$lib/InferenceWidget/InferenceWidget.svelte";
	import ModeSwitcher from "$lib/_demo/ModeSwitcher.svelte";
	import type { ModelData } from "../../../interfaces/Types";

	const models: ModelData[] = [
		{
			modelId: "distilbert-base-uncased",
			pipeline_tag: "fill-mask",
			mask_token: "[MASK]",
			widgetData: [{ text: "The goal of life is [MASK]." }],
		},
		{
			modelId: "dbmdz/bert-large-cased-finetuned-conll03-english",
			pipeline_tag: "token-classification",
			widgetData: [
				{
					text: "My name is Clara and I live in Berkeley, California. I work at this cool company called Hugging Face.",
				},
			],
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
				<a class="text-xs block mb-3 text-gray-300" href="/{model.modelId}">
					<code>{model.modelId}</code>
				</a>
				<div class="p-5 shadow-sm rounded-xl bg-white max-w-md">
					<InferenceWidget {model} />
				</div>
			</div>
		{/each}
	</div>
</div>
