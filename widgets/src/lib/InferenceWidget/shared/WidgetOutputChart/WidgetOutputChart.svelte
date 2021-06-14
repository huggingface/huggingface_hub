<script>
	export let classNames = "";
	export let output: Array<{ label: string; score: number }> = [];

	$: scoreMax = output.reduce(
		(acc, entry) => (entry.score > acc ? entry.score : acc),
		0
	);
</script>

{#if output.length}
	<div class="space-y-4 {classNames}">
		{#each output as { label, score }, index}
			<div
				class="flex items-start justify-between font-mono text-xs animate__animated animate__fadeIn leading-none"
				style="animation-delay: {0.04 * index}s"
			>
				<div class="flex-1">
					<div
						class="h-1 mb-1 rounded bg-gradient-to-r from-purple-400 to-purple-200"
						style={`width: ${Math.ceil((score / scoreMax) * 100 * 0.8)}%;`}
					/>
					<p>{label}</p>
				</div>
				<p class="pl-2">{score.toFixed(3)}</p>
			</div>
		{/each}
	</div>
{/if}
