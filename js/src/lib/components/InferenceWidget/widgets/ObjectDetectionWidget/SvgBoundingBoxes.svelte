<!-- 
for Tailwind:
text-red-400
text-green-400
text-yellow-400
text-blue-400
text-orange-400
text-purple-400
text-cyan-400
text-lime-400
 -->
<script>
	import type { DetectedObject } from "../../shared/types";

	import { afterUpdate } from "svelte";

	type Rect = { x: number; y: number; width: number; height: number };

	let containerEl: HTMLElement;
	let imgEl: HTMLImageElement;
	let wrapperHeight = 0;
	let wrapperWidth = 0;
	let boxes: Array<{
		color: string;
		index: number;
		rect: Rect;
	}> = [];

	export let classNames = "";
	export let highlightIndex = -1;
	export let imgSrc: string;
	export let output: DetectedObject[] = [];
	export let mouseover: (index: number) => void = () => {};
	export let mouseout: () => void = () => {};

	$: {
		if (imgEl?.naturalWidth && imgEl?.naturalHeight) {
			const widthScale = wrapperWidth / imgEl.naturalWidth;
			const heightScale = wrapperHeight / imgEl.naturalHeight;
			boxes = output
				.map((val, index) => ({ ...val, index }))
				.map(({ box, color, index }) => {
					const rect = {
						x: box.xmin * widthScale,
						y: box.ymin * heightScale,
						width: (box.xmax - box.xmin) * widthScale,
						height: (box.ymax - box.ymin) * heightScale,
					};
					return { rect, color, index };
				})
				.sort((a, b) => getArea(b.rect) - getArea(a.rect));
		}
	}

	function getArea(rect: Rect): number {
		return rect.width * rect.height;
	}

	afterUpdate(() => {
		wrapperWidth = containerEl.clientWidth;
		wrapperHeight = containerEl.clientHeight;
	});
</script>

<div
	class="relative top-0 left-0 inline-flex {classNames}"
	bind:this={containerEl}
>
	<div class="flex justify-center max-w-sm">
		<img
			alt=""
			class="relative top-0 left-0 object-contain"
			src={imgSrc}
			bind:this={imgEl}
		/>
	</div>

	<svg
		class="absolute top-0 left-0"
		viewBox={`0 0 ${wrapperWidth} ${wrapperHeight}`}
		xmlns="http://www.w3.org/2000/svg"
	>
		{#each boxes as { rect, color, index }}
			<rect
				class="transition duration-200 ease-in-out text-{color}-400 stroke-current fill-current"
				fill-opacity={highlightIndex === -1 || highlightIndex === index
					? "0.1"
					: "0.0"}
				opacity={highlightIndex === -1 || highlightIndex === index
					? "1"
					: "0.0"}
				{...rect}
				stroke-width="2"
				on:mouseover={() => mouseover(index)}
				on:mouseout={mouseout}
			/>
		{/each}
	</svg>
</div>
