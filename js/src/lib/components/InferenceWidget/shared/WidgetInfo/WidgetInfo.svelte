<script lang="ts">
	import type { LoadingStatus } from "../types";

	export let computeTime: string;
	export let error: string;
	export let modelStatus: LoadingStatus = "unknown";

	const status = {
		error: "⚠️ This model could not be loaded by the inference API. ⚠️",
		loaded: "This model is currently loaded and running on the Inference API.",
		unknown: "This model can be loaded on the Inference API on-demand.",
	} as const;
</script>

<div class="mt-2">
	<div class="text-gray-400 text-xs">
		{#if computeTime}
			Computation time on cpu: {computeTime}
		{:else if modelStatus}
			{status[modelStatus]}
		{/if}
	</div>
	{#if error}
		<div class="alert alert-error mt-3">{error}</div>
	{/if}
</div>
