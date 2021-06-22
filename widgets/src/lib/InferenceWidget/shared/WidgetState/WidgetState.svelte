<script>
	export let currentState: "error" | "loaded" | "unknown" | undefined;
</script>

<div class="blankslate">
	<div class="subtitle text-xs text-gray-500">
		<div class="mt-2 loaded {currentState !== 'loaded' ? 'hidden' : ''}">
			This model is currently loaded and running on the Inference API.
		</div>
		<div class="mt-2 error {currentState !== 'error' ? 'hidden' : ''}">
			⚠️ This model could not be loaded by the inference API. ⚠️
		</div>
		<div class="mt-2 unknown {currentState !== 'unknown' ? 'hidden' : ''}">
			This model can be loaded on the Inference API on-demand.
		</div>
	</div>
</div>
