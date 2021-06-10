<script context="module">
	import type { Load } from "@sveltejs/kit";
	import type { ModelData } from "../../../interfaces/Types";

	import InferenceWidget from "$lib/InferenceWidget/InferenceWidget.svelte";

	export const load: Load = async ({ page, fetch }) => {
		const url = `https://huggingface.co/api/models/${page.params.model}`;
		const model = await (await fetch(url)).json();
		return {
			props: {
				model,
			},
		};
	};
</script>

<script>
	export let model: ModelData;
</script>

<div class="py-24 min-h-screen bg-gray-50 dark:bg-gray-900">
	<div class="container">
		<div>
			<a class="text-xs block mb-3 text-gray-300" href="/{model.modelId}">
				<code>{model.modelId}</code>
			</a>
			<div class="p-5 shadow-sm rounded-xl bg-white max-w-md">
				<InferenceWidget {model} />
			</div>
		</div>

		<pre
			class="text-xs text-gray-500 px-3 py-4 mt-16">
			{ JSON.stringify(model, null, 2) }
		</pre>
	</div>
</div>
