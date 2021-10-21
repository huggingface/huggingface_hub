<script>
	import type { ImageSegment } from "../../shared/types";
	import { afterUpdate } from "svelte";

	export let classNames = "";
	export let highlightIndex: number;
	export let imgSrc = "";
	export let mousemove: (e: Event, canvasW: number, canvasH: number) => void =
		() => {};
	export let mouseout: () => void = () => {};
	export let output: ImageSegment[] = [];

	let containerEl: HTMLElement;
	let canvas: HTMLCanvasElement;
	let imgEl: HTMLImageElement;
	let width = 0;
	let height = 0;
	let startTs: DOMHighResTimeStamp;

	const animDuration = 200;

	function draw() {
		width = containerEl.clientWidth;
		height = containerEl.clientHeight;
		console.log("Update called", { width, height });
		startTs = performance.now();
		darwHelper();
	}

	function darwHelper() {
		const maskToDraw = output.reduce((arr, o, i) => {
			const mask = o?.bitmap;
			if (mask && (i === highlightIndex || highlightIndex === -1)) {
				arr.push(mask);
			}
			return arr;
		}, []);

		const ctx = canvas?.getContext("2d");

		if (ctx && maskToDraw.length) {
			const duration = performance.now() - startTs;
			ctx.globalAlpha = Math.min(duration, animDuration) / animDuration;
			ctx.drawImage(imgEl, 0, 0, width, height);
			for (const mask of maskToDraw) {
				ctx.drawImage(mask, 0, 0, width, height);
			}
			if (duration < animDuration) {
				// when using canvas, prefer to use requestAnimationFrame over setTimeout & setInterval
				// https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame
				window.requestAnimationFrame(darwHelper);
			}
		}
	}

	afterUpdate(draw);
</script>

<svelte:window on:resize={draw} />

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
	<canvas
		class="absolute top-0 left-0"
		{width}
		{height}
		bind:this={canvas}
		on:mousemove={(e) => mousemove(e, width, height)}
		on:mouseout={mouseout}
	/>
</div>
