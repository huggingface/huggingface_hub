<script context="module">
	import type { Load } from "@sveltejs/kit";
	import type { ModelData } from "../lib/interfaces/Types";

	import InferenceWidget from "../lib/components/InferenceWidget/InferenceWidget.svelte";

	/**
	 * This page is capable of loading any model
	 * on the fly and representing its widget.
	 *
	 * If model doesn't exist, it will display an error message.
	 */

	export const load: Load = async ({ page, fetch }) => {
		const url = `https://huggingface.co/api/models/${page.params.model}`;
		try {
			const model = await (await fetch(url)).json();
			return {
				props: {
					model,
				},
			};
		} catch {
			return {
				props: {
					message: `Error. Probably non existent model.`,
				},
			};
		}
	};
</script>

<script>
	import ModeSwitcher from "../lib/components/DemoThemeSwitcher/DemoThemeSwitcher.svelte";

	export let model: ModelData | undefined;
	export let message: string | undefined;
</script>

<a href="/" class="pt-3 ml-3 block underline">← Back to index</a>
<ModeSwitcher />

<div class="container py-24">
	{#if model}
		<div>
			<a class="text-xs block mb-3 text-gray-300" href="/{model.id}">
				<code>{model.id}</code>
			</a>
			<div class="p-5 shadow-sm rounded-xl bg-white max-w-3xl">
				<InferenceWidget {model} />
			</div>
		</div>

		<pre
			class="text-xs text-gray-900 px-3 py-4 mt-16">
			{ JSON.stringify(model, null, 2) }
		</pre>
	{:else}
		<div>Error. Probably non existent model. {message}</div>
	{/if}
</div>
