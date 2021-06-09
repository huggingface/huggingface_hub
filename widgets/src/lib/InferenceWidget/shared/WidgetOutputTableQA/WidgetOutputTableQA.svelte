<script>
	import { fly } from "svelte/transition";

	interface Output {
		aggregator?: string;
		answer: string;
		coordinates: [number, number][];
		cells: number[];
	}

	export let output: Output;
</script>

<div
	class="col-span-12 overflow-x-auto px-3 h-10 border border-b-0 flex items-center bg-gradient-to-r to-white rounded-t-lg {output
		.cells.length
		? 'border-green-50 from-green-50 via-green'
		: 'border-red-50 from-red-50 via-red '}"
	in:fly
>
	<span class="whitespace-nowrap">
		{#if output.cells.length}
			{output.cells.length}
			match{output.cells.length > 1 ? "es" : ""}
			:
		{:else}
			No matches
		{/if}
	</span>
	{#if output.cells.length}
		{#each output.cells as answer}
			<span
				class="whitespace-nowrap bg-green-100 border border-green-200 text-green-800 px-1 leading-tight rounded ml-2"
				>{answer}</span
			>
		{/each}
		{#if output.aggregator !== "NONE"}
			<span
				class="whitespace-nowrap ml-auto bg-blue-100 border border-blue-200 text-blue-800 px-1 leading-tight rounded"
				>{output.aggregator}</span
			>
		{/if}
	{/if}
</div>
