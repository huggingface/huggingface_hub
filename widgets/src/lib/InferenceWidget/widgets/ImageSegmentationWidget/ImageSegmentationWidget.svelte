<script>
	import type { WidgetProps } from "../../shared/types";

	import * as tailwindColors from "tailwindcss/colors";
	import WidgetCanvas from "../../shared/WidgetCanvas/WidgetCanvas.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";
	import { onMount } from "svelte";
	import { width, heigth, canvas, img } from "../../shared/WidgetCanvas/stores";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	let computeTime = "";
	let error: string = "";
	let fileInput: HTMLInputElement;
	let isLoading = false;
	let imgSrc = "";
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Array<{
		label: string;
		score: number;
		mask: any;
		color?: string;
	}> = []; //TODO: define mask type
	let outputJson: string;
	let maskImages = []; //TODO: define type
	let highlightIndex = -1;
	let maskW: number;
	let maskH: number;
	let ctx: CanvasRenderingContext2D;
	let updateCounter = 1;
	const colors = generateColors();
	$: maskW = output?.[0]?.["mask"].length ?? 0;
	$: maskH = output?.[0]?.["mask"]?.[0].length ?? 0;
	$: ctx = $canvas?.getContext("2d");
	$: maskImages = output.map((row, index) => {
		row.color = colors.next().value as string;
		const maskImage = getMaskImage(index);
		return maskImage;
	});
	$: {
		if ($canvas && $width && $heigth && maskImages && updateCounter) {
			drawCanvas();
		}
	}

	function onSelectFile() {
		const file = fileInput.files?.[0];
		if (file) {
			imgSrc = URL.createObjectURL(file);
			getOutput(file);
		}
	}

	async function getOutput(file: File, withModelLoading = false) {
		if (!file) {
			return;
		}

		const requestBody = { file };

		isLoading = true;

		const res = await getResponse(
			apiUrl,
			model.modelId,
			requestBody,
			apiToken,
			parseOutput,
			withModelLoading
		);

		isLoading = false;
		// Reset values
		computeTime = "";
		error = "";
		modelLoading = { isLoading: false, estimatedTime: 0 };
		output = [];
		outputJson = "";

		if (res.status === "success") {
			computeTime = res.computeTime;
			output = res.output;
			// outputJson = res.outputJson;
		} else if (res.status === "loading-model") {
			modelLoading = {
				isLoading: true,
				estimatedTime: res.estimatedTime,
			};
			getOutput(file, true);
		} else if (res.status === "error") {
			error = res.error;
		}
	}

	function isValidOutput(
		arg: any
	): arg is { label: string; score: number; mask: any }[] {
		return (
			Array.isArray(arg) &&
			arg.every(
				(x) => typeof x.label === "string" && typeof x.score === "number"
				// TODO: check mask type
			)
		);
	}

	function parseOutput(
		body: unknown
	): Array<{ label: string; score: number; mask: any }> {
		return isValidOutput(body) ? body : [];
	}

	function drawCanvas() {
		ctx.drawImage($img, 0, 0, $width, $heigth);
		maskImages.map(async (maskImage, i) => {
			if (highlightIndex !== -1 && highlightIndex !== i) return;
			const maskBitMap = await createImageBitmap(maskImage);
			ctx.drawImage(maskBitMap, 0, 0, $width, $heigth);
		});
	}

	function clamp(n: number, min: number, max: number): number {
		return Math.max(min, Math.min(n, max));
	}

	function mousemove(e): void {
		if ($canvas && output.length) {
			let { layerX, layerY } = e;
			layerX = clamp(layerX, 0, $width);
			layerY = clamp(layerY, 0, $heigth);
			const row = Math.floor((layerX / $width) * maskH);
			const col = Math.floor((layerY / $heigth) * maskW);
			let index = -1;
			for (let [i, val] of output.entries()) {
				const { mask } = val;
				if (mask[col][row]) {
					index = i;
					break;
				}
			}
			if (highlightIndex !== index) mouseover(index);
		}
	}

	function mouseout(): void {
		highlightIndex = -1;
		updateCounter++;
	}

	function mouseover(index: number): void {
		highlightIndex = index;
		updateCounter++;
	}

	function* generateColors(): Generator<string> {
		let index = 0;
		const colorKeys = Object.keys(tailwindColors)
			.filter(
				(key) => !key.match(/black|white|gray|emerald|green|orange|purple/i)
			)
			.sort(() => Math.random() - 0.5);
		while (true) {
			const color = colorKeys[index % colorKeys.length];
			yield color;
			index++;
		}
	}

	// TODO: subject to modification depending on API output's mask format
	function getMaskImage(index: number): ImageData {
		const { mask, color } = output[index];
		const maskFlat = mask.flat();
		const maskImage = ctx.createImageData(maskH, maskW);
		const colorHex = tailwindColors[color][400];
		const [r, g, b] = colorHex.match(/\w\w/g).map((x) => parseInt(x, 16));
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

	onMount(() => {
		getOutput(null, null);
	});
</script>

<WidgetWrapper
	{apiUrl}
	{computeTime}
	{error}
	{model}
	{modelLoading}
	{noTitle}
	{outputJson}
>
	<svelte:fragment slot="top">
		<form>
			<WidgetDropzone
				{isLoading}
				bind:fileInput
				onChange={onSelectFile}
				{imgSrc}
				innerWidget={WidgetCanvas}
				innerWidgetProps={{ imgSrc, mousemove, mouseout }}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetOutputChart
			classNames="mt-4"
			{output}
			{highlightIndex}
			{mouseover}
			{mouseout}
		/>
	</svelte:fragment>
</WidgetWrapper>
