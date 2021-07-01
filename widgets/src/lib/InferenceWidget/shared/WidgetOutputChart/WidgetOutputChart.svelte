<!-- 
	hack for tailwind pruning
from-rose-400 to-rose-200
from-pink-400 to-pink-200
from-fuchsia-400 to-fuchsia-200
from-purple-400 to-purple-200
from-violet-400 to-violet-200
from-indigo-400 to-indigo-200
from-blue-400 to-blue-200
from-lightBlue-400 to-lightBlue-200
from-cyan-400 to-cyan-200
from-teal-400 to-teal-200
from-green-400 to-green-200
from-lime-400 to-lime-200
from-yellow-400 to-yellow-200
from-amber-400 to-amber-200
from-orange-400 to-orange-200
from-red-400 to-red-200
 -->
<script>
	export let classNames = "";
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
	<div class="space-y-4 {classNames}">
		{#each output as { label, score, color }, index}
			<div
				class="flex items-start justify-between font-mono text-xs animate__animated animate__fadeIn leading-none 
					{highlightIndex !== -1 && highlightIndex !== index
					? 'opacity-30 filter grayscale'
					: ''}
				"
				style="animation-delay: {0.04 * index}s"
				on:mouseover={() => mouseover(index)}
				on:mouseout={mouseout}
			>
				<div class="flex-1">
					<div
						class="h-1 mb-1 rounded bg-gradient-to-r from-{color ??
							'purple'}-400 to-{color ?? 'purple'}-200"
						style={`width: ${Math.ceil((score / scoreMax) * 100 * 0.8)}%;`}
					/>
					<p>{label}</p>
				</div>
				<p class="pl-2">{score.toFixed(3)}</p>
			</div>
		{/each}
	</div>
{/if}
