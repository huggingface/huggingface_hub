<script>
	import { clip } from "../../shared/ViewUtils";
	import * as tailwindColors from "tailwindcss/colors";
	import { highlightIndex, updateCounter } from "./stores";
	export let output: any;
	export let src = "";
	export let mouseover: (index: number) => void = () => {};
	export let mouseout: () => void = () => {};

	let canvas: HTMLCanvasElement;
	let canvasInterval: ReturnType<typeof setInterval>;
	let ctx: CanvasRenderingContext2D;
	let img: HTMLImageElement;
	let width = 0;
	let height = 0;
	let maskW = 0;
	let maskH = 0;
	
	$: ctx = canvas?.getContext("2d");
	$: maskW = output?.[0]?.["mask"].length ?? 0;
	$: maskH = output?.[0]?.["mask"]?.[0].length ?? 0;
	$: maskImages = output.map((_, index: number) => getMaskImage(index));
	$: {
		if (canvas && height && maskImages && $updateCounter) {
			drawCanvas();
		}
	}

	function drawCanvas() {
		let alpha = 0.05;
		clearInterval(canvasInterval);
		canvasInterval = setInterval(() => {
			alpha += 0.05;
			ctx.globalAlpha = alpha;
			ctx.drawImage(img, 0, 0, width, height);
			maskImages.map(async (maskImage, i) => {
				if ($highlightIndex !== -1 && $highlightIndex !== i) return;
				const maskBitMap = await createImageBitmap(maskImage);
				ctx.drawImage(maskBitMap, 0, 0, width, height);
			});
			if (alpha >= 1.0) {
				clearInterval(canvasInterval);
			}
		}, 10);
	}

	// TODO: subject to modification depending on API output's mask format
	function getMaskImage(index: number): ImageData {
		const { mask, color } = output[index];
		const maskFlat = mask.flat();
		const maskImage = ctx.createImageData(maskH, maskW);
		const [r, g, b]: number[] = tailwindColors[color][400]
			.match(/\w\w/g)
			.map((x) => parseInt(x, 16));
		for (let i = 0; i < maskImage.data.length; i += 4) {
			const val = maskFlat[i / 4];
			if (val) {
				maskImage.data[i + 0] = r; // R value
				maskImage.data[i + 1] = g; // G value
				maskImage.data[i + 2] = b; // B value
				maskImage.data[i + 3] = Math.floor(255 * 0.7); // A value
			}
		}
		return maskImage;
	}

	function mousemove(e): void {
		if (canvas && output.length) {
			let { layerX, layerY } = e;
			layerX = clip(layerX, 0, width);
			layerY = clip(layerY, 0, height);
			const row = Math.floor((layerX / width) * maskH);
			const col = Math.floor((layerY / height) * maskW);
			let index = -1;
			for (let [i, val] of output.entries()) {
				const { mask } = val;
				if (mask[col][row]) {
					index = i;
					break;
				}
			}
			if ($highlightIndex !== index) {
				$highlightIndex = index;
				mouseover(index);
			}
		}
	}
</script>

<div
	class="relative top-0 left-0"
	bind:clientWidth={width}
	bind:clientHeight={height}
>
	<img alt="" class="relative top-0 left-0" {src} bind:this={img} />
	<canvas
		class="absolute top-0 left-0"
		{width}
		{height}
		bind:this={canvas}
		on:mousemove={mousemove}
		on:mouseout={mouseout}
	/>
</div>
