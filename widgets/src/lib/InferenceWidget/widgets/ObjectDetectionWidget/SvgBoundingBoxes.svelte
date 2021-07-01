<script>
	export let width = 0;
	export let height = 0;
	export let src = "";
	export let highlightIndex = -1;
	export let output: Array<{ boundingBox: any; color: string }> = [];
	export let mouseover: (index: number) => void = () => {};
	export let mouseout: () => void = () => {};

	// TODO: define boxes type
	$: boxes = output
		.map((val, index) => ({ ...val, index }))
		.sort((a, b) => getArea(b.boundingBox) - getArea(a.boundingBox))
		.map(({ boundingBox, color, index }) => {
			const vertices = boundingBox.map(({ x, y }) => {
				x *= width;
				y *= height;
				return `${x},${y}`;
			});
			const points = `${vertices.join(" ")}`;
			return { points, color, index };
		});

	// TODO: define boundingBox type
	function getArea(boundingBox: any): number {
		const corner1 = boundingBox[0];
		const corner3 = boundingBox[2];
		return Math.abs(corner1.x - corner3.x) * Math.abs(corner1.y - corner3.y);
	}
</script>

<div
	class="relative top-0 left-0"
	bind:clientWidth={width}
	bind:clientHeight={height}
>
	<img alt="" class="relative top-0 left-0" {src} />

	<svg
		class="absolute top-0 left-0"
		viewBox={`0 0 ${width} ${height}`}
		xmlns="http://www.w3.org/2000/svg"
	>
		{#each boxes as { points, color, index }}
			<polygon
				{points}
				stroke={color}
				fill={color}
				stroke-width="3"
				opacity={highlightIndex === -1 || highlightIndex === index
					? "1"
					: "0.0"}
				fill-opacity={highlightIndex === -1 || highlightIndex === index
					? "0.15"
					: "0.0"}
				class="transition duration-200 ease-in-out"
				on:mouseover={() => mouseover(index)}
				on:mouseout={mouseout}
			/>
		{/each}
	</svg>
</div>
