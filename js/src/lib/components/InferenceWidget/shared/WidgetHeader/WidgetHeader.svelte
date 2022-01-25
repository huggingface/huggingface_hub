<script lang="ts">
	import type { PipelineType } from "../../../../interfaces/Types";

	import { getPipelineTask } from "../../../../utils/ViewUtils";
	import { TASKS_DATA } from "../../../../../../../tasks/src/tasksData";
	import IconInfo from "../../../Icons/IconInfo.svelte";
	import IconLightning from "../../../Icons/IconLightning.svelte";
	import ModelPipelineTag from "../../../ModelPipelineTag/ModelPipelineTag.svelte";

	export let noTitle = false;
	export let pipeline: keyof typeof PipelineType | undefined;

	$: task = pipeline ? getPipelineTask(pipeline) : undefined;
</script>

<div class="font-semibold flex items-center mb-2">
	{#if !noTitle}
		<div class="text-lg flex items-center">
			<IconLightning classNames="-ml-1 mr-1 text-yellow-500" />
			Hosted inference API
		</div>
		<a target="_blank" href="https://api-inference.huggingface.co/">
			<IconInfo classNames="ml-1.5 text-sm text-gray-400 hover:text-black" />
		</a>
	{/if}
</div>
<div
	class="flex items-center justify-between flex-wrap w-full max-w-full text-sm text-gray-500 mb-0.5"
>
	{#if pipeline}
		<a
			class={TASKS_DATA[task] ? "hover:underline" : undefined}
			href={TASKS_DATA[task] ? `/tasks/${task}` : undefined}
			target="_blank"
			title={TASKS_DATA[task] ? `Learn more about ${task}` : undefined}
		>
			<ModelPipelineTag classNames="mr-2 mb-1.5" {pipeline} />
		</a>
	{/if}
	<slot />
</div>
