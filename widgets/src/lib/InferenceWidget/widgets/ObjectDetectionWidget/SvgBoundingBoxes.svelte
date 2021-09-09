<script>
	import type { Box } from "../../shared/types";
	import * as colors from "tailwindcss/colors";
	let height = 0;
	let width = 0;
	export let imgSrc = "";
	export let highlightIndex = -1;
	export let output: Array<{ box: Box; color: string }> = [];
	export let mouseover: (index: number) => void = () => {};
	export let mouseout: () => void = () => {};

	$: boxes = output
		.map((val, index) => ({ ...val, index }))
		.sort((a, b) => getArea(b.box) - getArea(a.box))
		.map(({ box, color, index }) => {
			const rect = {
				x: box.xmin,
				y: box.ymin,
				width: box.xmax - box.xmin,
				height: box.ymax - box.ymin,
			};
			return { rect, color, index };
		});

	function getArea(box: Box): number {
		return (box.xmax - box.xmin) * (box.ymax - box.ymin);
	}
</script>

{#if imgSrc}
	<div
		class="relative top-0 left-0 inline-flex"
		bind:clientWidth={width}
		bind:clientHeight={height}
	>
		<img alt="" class="relative top-0 left-0" src={imgSrc} />

		<svg
			class="absolute top-0 left-0"
			viewBox={`0 0 ${width} ${height}`}
			xmlns="http://www.w3.org/2000/svg"
		>
			{#each boxes as { rect, color, index }}
				<rect
					{...rect}
					stroke={colors[color][400]}
					fill={colors[color][400]}
					stroke-width="2"
					opacity={highlightIndex === -1 || highlightIndex === index
						? "1"
						: "0.0"}
					fill-opacity={highlightIndex === -1 || highlightIndex === index
						? "0.1"
						: "0.0"}
					class="transition duration-200 ease-in-out"
					on:mouseover={() => mouseover(index)}
					on:mouseout={mouseout}
				/>
			{/each}
		</svg>
	</div>
{/if}
