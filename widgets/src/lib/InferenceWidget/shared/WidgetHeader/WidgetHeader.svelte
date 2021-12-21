<script lang="ts">
	import Icon from "../../../Icon/Icon.svelte";
	import IconInfo from "../../../Icon/data/IconInfo";
	import IconLightning from "../../../Icon/data/IconLightning";
	import ModelPipelineTag from "../../../ModelPipelineTag/ModelPipelineTag.svelte";

	export let noTitle = false;
	export let pipeline: string | undefined;
</script>

<div class="font-semibold flex items-center mb-2">
	{#if !noTitle}
		<div class="text-lg flex items-center">
			<Icon data={IconLightning} classNames="-ml-1 mr-1 text-yellow-500" />
			Hosted inference API
		</div>
		<a target="_blank" href="/docs">
			<Icon
				data={IconInfo}
				classNames="ml-1.5 text-sm text-gray-400 hover:text-black"
			/>
		</a>
	{/if}
</div>
<div class="flex items-center text-sm text-gray-500 mb-1.5">
	{#if pipeline}
		<ModelPipelineTag {pipeline} />
	{/if}
	<div class="ml-auto">
		<slot />
	</div>
</div>
