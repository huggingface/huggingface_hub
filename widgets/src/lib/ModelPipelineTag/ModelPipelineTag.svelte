<script>
	import { PipelineType } from "$lib/interfaces/Types";
	import ModelPipelineIcon from "../ModelPipelineIcon/ModelPipelineIcon.svelte";

	export let pipeline = "";
</script>

<div class="inline-flex items-center">
	<ModelPipelineIcon classNames="mr-1" {pipeline} />
	<span>
		{PipelineType[pipeline] ?? pipeline}
	</span>
</div>
