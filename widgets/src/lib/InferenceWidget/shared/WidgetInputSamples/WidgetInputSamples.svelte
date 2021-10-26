<script>
	export let inputSamples: Record<string, any>[];
	export let applyInputSample: (sample: Record<string, any>[]) => void;

	function onChange(e: Event) {
		const { value } = e.target as HTMLOptionElement;
		const sample = inputSamples[value];
		applyInputSample(sample);
	}
</script>

<select
	class="border-none text-sm w-28 truncate dark:bg-gray-950"
	on:change={onChange}
>
	<option selected disabled>Examples</option>
	{#each inputSamples as { example_title }, i}
		<option value={i}>{example_title}</option>
	{/each}
</select>
