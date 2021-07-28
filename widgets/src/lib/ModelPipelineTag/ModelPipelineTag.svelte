<script>
	import { PIPELINE_TYPE_PRETTY_NAMES } from "$lib/interfaces/Types";
	import ModelPipelineIcon from "../ModelPipelineIcon/ModelPipelineIcon.svelte";

	export let pipeline = "";
</script>

<div class="inline-flex items-center">
	<ModelPipelineIcon classNames="mr-1" {pipeline} />
	<span>
		{PIPELINE_TYPE_PRETTY_NAMES[pipeline] ?? pipeline}
	</span>
</div>
