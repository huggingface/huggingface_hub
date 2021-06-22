<script>
	export let autoplay = false;
	export let classNames = "";
	export let controls = true;
	export let label = "";
	export let src: string;
</script>

<div class={classNames}>
	{#if $$slots.default}
		<slot />
	{:else if label.length}
		<div class="mb-1.5 text-sm text-gray-500 truncate">{label}</div>
	{/if}
	<!-- svelte-ignore a11y-media-has-caption -->
	<audio {autoplay} {controls} {src} />
</div>
