<script>
	import InferenceWidget from "../lib/components/InferenceWidget/InferenceWidget.svelte";
	import ModeSwitcher from "../lib/components/DemoThemeSwitcher/DemoThemeSwitcher.svelte";
	import type { ModelData } from "../lib/interfaces/Types";

	const models: ModelData[] = [
		{
			id: "ThomasSimonini/ppo-LunarLander-v2",
			pipeline_tag: "reinforcement-learning",
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
