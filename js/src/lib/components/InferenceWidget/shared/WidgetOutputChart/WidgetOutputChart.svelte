<!-- 
for Tailwind:
from-blue-400 to-blue-200 dark:from-blue-400 dark:to-blue-600
from-cyan-400 to-cyan-200 dark:from-cyan-400 dark:to-cyan-600
from-green-400 to-green-200 dark:from-green-400 dark:to-green-600
from-indigo-400 to-indigo-200 dark:from-indigo-400 dark:to-indigo-600
from-lime-400 to-lime-200 dark:from-lime-400 dark:to-lime-600
from-orange-400 to-orange-200 dark:from-orange-400 dark:to-orange-600
from-purple-400 to-purple-200 dark:from-purple-400 dark:to-purple-600
from-red-400 to-red-200 dark:from-red-400 dark:to-red-600
from-yellow-400 to-yellow-200 dark:from-yellow-400 dark:to-yellow-600
 -->
<script>
	export let classNames = "";
	export let defaultBarColor = "purple";
	export let output: Array<{ label: string; score: number; color?: string }> =
		[];
	export let highlightIndex = -1;
	export let mouseover: (index: number) => void = () => {};
	export let mouseout: () => void = () => {};

	$: scoreMax = output.reduce(
		(acc, entry) => (entry.score > acc ? entry.score : acc),
		0
	);
</script>

{#if output.length}
	<div class="space-y-3.5 {classNames}">
		<!-- NB: We sadly can't do color = defaultBarColor as the Svelte compiler will throw an unused-export-let warning (bug  on their side) ... -->
		{#each output as { label, score, color }, index}
			<div
				class="flex items-start justify-between font-mono text-xs leading-none
					animate__animated animate__fadeIn transition duration-200 ease-in-out
					{highlightIndex !== -1 &&
					highlightIndex !== index &&
					'opacity-30 filter grayscale'}
				"
				style="animation-delay: {0.04 * index}s"
				on:mouseover={() => mouseover(index)}
				on:mouseout={mouseout}
			>
				<div class="flex-1">
					<div
						class="h-1 mb-1 rounded bg-gradient-to-r 
							from-{color ?? defaultBarColor}-400 
							to-{color ?? defaultBarColor}-200 
							dark:from-{color ?? defaultBarColor}-400 
							dark:to-{color ?? defaultBarColor}-600"
						style={`width: ${Math.ceil((score / scoreMax) * 100 * 0.8)}%;`}
					/>
					<span class="leading-snug">{label}</span>
				</div>
				<span class="pl-2">{score.toFixed(3)}</span>
			</div>
		{/each}
	</div>
{/if}
