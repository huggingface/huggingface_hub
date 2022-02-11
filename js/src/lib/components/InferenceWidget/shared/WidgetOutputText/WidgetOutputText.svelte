<script>
	export let classNames: string;
	export let output: string;
</script>

{#if output.length}
	<p class="alert alert-success {classNames}">
		{output}
	</p>
{/if}
