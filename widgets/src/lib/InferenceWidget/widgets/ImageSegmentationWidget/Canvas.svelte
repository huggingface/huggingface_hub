<script>
	import type { ImageSegment } from "../../shared/types";
	import { afterUpdate } from "svelte";

	export let classNames = "";
	export let highlightIndex: number;
	export let imgSrc = "";
	export let mousemove: (e: Event, canvasW: number, canvasH: number) => void =
		() => {};
	export let mouseout: () => void = () => {};
	export let output: ImageSegment[];

	let canvas: HTMLCanvasElement;
	let imgEl: HTMLImageElement;
	let width = 0;
	let height = 0;
	let startTs: DOMHighResTimeStamp;

	afterUpdate(() => {
		startTs = performance.now();
		draw();
	});

	function draw() {
		if (!output) {
			return;
		}
		const animDuration = 200;
		const masks = [];
		if (highlightIndex === -1) {
			for (const o of output) {
				const mask = o?.bitmap;
				if (mask) {
					masks.push(mask);
				}
			}
		} else {
			const mask = output?.[highlightIndex]?.bitmap;
			if (mask) {
				masks.push(mask);
			}
		}
		const ctx = canvas?.getContext("2d");
		if (ctx && masks.length) {
			const duration = performance.now() - startTs;
			ctx.globalAlpha = Math.min(duration, animDuration) / animDuration;
			ctx.drawImage(imgEl, 0, 0, width, height);
			for (const mask of masks) {
				ctx.drawImage(mask, 0, 0, width, height);
			}
			if (duration < animDuration) {
				// when using canvas, prefer to use requestAnimationFrame over setTimeout & setInterval
				// https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame
				window.requestAnimationFrame(draw);
			}
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
