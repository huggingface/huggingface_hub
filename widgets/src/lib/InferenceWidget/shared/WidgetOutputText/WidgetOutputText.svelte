<script>
	export let classNames: string;
	export let output: string;

	$: isCode = output?.includes("\n");
</script>

{#if output.length}
	{#if isCode}
		<pre class="alert alert-success overflow-auto {classNames}">
			{output}
		</pre>
	{:else}
		<p class="alert alert-success {classNames}">
			{output}
		</p>
	{/if}
{/if}
