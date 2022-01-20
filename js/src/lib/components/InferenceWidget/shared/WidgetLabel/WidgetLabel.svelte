<script>
	export let classNames = "";
	export let forInput: string | undefined = undefined;
	export let label: string | undefined;
</script>

<label class="block {classNames}" for={forInput}>
	<slot name="before" />
	{#if label}
		<span class="text-sm text-gray-500">{label}</span>
	{/if}
	<slot name="after" />
</label>
