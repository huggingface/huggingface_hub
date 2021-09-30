<script>
	import { afterUpdate } from "svelte";

	export let bitmaps: ImageBitmap[] = [];
	export let classNames = "";
	export let highlightIndex = -1;
	export let imgSrc = "";
	export let mousemove: (e: Event, canvasW: number, canvasH: number) => void =
		() => {};
	export let mouseout: () => void = () => {};

	let canvas: HTMLCanvasElement;
	let imgEl: HTMLImageElement;
	let width = 0;
	let height = 0;

	afterUpdate(() => {
		draw();
	});

	function draw() {
		const bitmap = bitmaps?.[highlightIndex];
		const ctx = canvas?.getContext("2d");
		if (bitmap && ctx) {
			ctx.drawImage(imgEl, 0, 0, width, height);
			ctx.drawImage(bitmap, 0, 0, width, height);
		}
	}
</script>

<div
	class="relative top-0 left-0 inline-flex {classNames}"
	bind:clientWidth={width}
	bind:clientHeight={height}
>
	<div class="flex justify-center max-w-sm">
		<img
			alt=""
			class="relative top-0 left-0 object-contain"
			src={imgSrc}
			bind:this={imgEl}
		/>
	</div>
	<canvas
		class="absolute top-0 left-0"
		{width}
		{height}
		bind:this={canvas}
		on:mousemove={(e) => mousemove(e, width, height)}
		on:mouseout={mouseout}
	/>
</div>
