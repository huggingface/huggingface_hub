<script lang="ts">
	import { PipelineType } from "../../interfaces/Types";
	import ModelPipelineIcon from "../ModelPipelineIcon/ModelPipelineIcon.svelte";

	export let classNames = "";
	export let pipeline = "";
</script>

<a
	class="inline-flex items-center hover:underline {classNames}"
	href="/tasks/{pipeline}"
	target="_blank"
	title="Learn more about {PipelineType[pipeline]}"
>
	<ModelPipelineIcon classNames="mr-1" {pipeline} />
	<span>
		{PipelineType[pipeline] ?? pipeline}
	</span>
</a>
