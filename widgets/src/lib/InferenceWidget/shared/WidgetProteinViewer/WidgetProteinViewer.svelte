<script>
	import { onMount } from "svelte";
	export let autoplay = false;
	export let classNames = "";
	export let controls = true;
	export let label = "";
	export let src: string;
	export let pdb_src: string;
	// create a `stage` object

	onMount(async () => {
		const NGL = await import("ngl");
		var stage = new NGL.Stage("viewport");
		// load a PDB structure and consume the returned `Promise`
		pdb_src = src;
		stage.loadFile(src).then(function (component) {
			// add a "cartoon" representation to the structure component
			component.addRepresentation("cartoon");
			// provide a "good" view of the structure
			component.autoView();
		});
	});
</script>

<div class={classNames}>
	{#if $$slots.default}
		<slot />
	{:else if label.length}
		<div class="mb-1.5 text-sm text-gray-500 truncate">{label}</div>
	{/if}
	<!-- svelte-ignore a11y-media-has-caption -->
	<div id="viewport" style="height:400px;width:400px" />
	<a href={pdb_src} class="btn-widget">Download pdb file</a>
</div>
