<script>
	import type { WidgetProps, ImageSegment } from "../../shared/types";
	import { onMount } from "svelte";
	import { clip, mod, COLORS } from "../../shared/ViewUtils";
	import { getResponse } from "../../shared/helpers";

	import Canvas from "./Canvas.svelte";
	import WidgetFileInput from "../../shared/WidgetFileInput/WidgetFileInput.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	const maskOpacity = Math.floor(255 * 0.6);

	let computeTime = "";
	let error: string = "";
	let highlightIndex = -1;
	let isLoading = false;
	let imgSrc = "";
	let imgW = 0;
	let imgH = 0;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: ImageSegment[];
	let outputJson: string;
	let warning: string = "";

	function onSelectFile(file: File | Blob) {
		imgSrc = URL.createObjectURL(file);
		getOutput(file);
	}

	async function getOutput(file: File | Blob, withModelLoading = false) {
		// TODO: remove demo api simlation
		isLoading = true;
		imgSrc = "./cats.jpg";
		const response = await fetch("./output.json");
		output = await response.json();
		addOutputColor(output);
		await Promise.all(output.map((o) => addOutputCanvasData(o)));
		warning = "Inferece API WIP: demo image is loaded";
		isLoading = false;

		// if (!file) {
		// 	return;
		// }

		// // Reset values
		// computeTime = "";
		// error = "";
		// warning = "";
		// output = null;
		// outputJson = "";

		// const requestBody = { file };

		// isLoading = true;

		// const res = await getResponse(
		// 	apiUrl,
		// 	model.modelId,
		// 	requestBody,
		// 	apiToken,
		// 	parseOutput,
		// 	withModelLoading
		// );

		// isLoading = false;
		// modelLoading = { isLoading: false, estimatedTime: 0 };

		// if (res.status === "success") {
		// 	computeTime = res.computeTime;
		// 	output = res.output;
		// 	if (output.segments_info.length === 0) {
		// 		warning = "No object was detected";
		// 	} else {
		// 		output = addOutputColor(output.segments_info);
		// 	}
		// 	// outputJson = res.outputJson;
		// } else if (res.status === "loading-model") {
		// 	modelLoading = {
		// 		isLoading: true,
		// 		estimatedTime: res.estimatedTime,
		// 	};
		// 	getOutput(file, true);
		// } else if (res.status === "error") {
		// 	error = res.error;
		// }
	}

	function isValidOutput(arg: any): arg is ImageSegment[] {
		return (
			Array.isArray(arg) &&
			arg.every(
				(x) =>
					typeof x.label === "string" &&
					typeof x.score === "number" &&
					typeof x.mask === "string"
			)
		);
	}
	function parseOutput(body: unknown): ImageSegment[] {
		if (isValidOutput(body)) {
			return body;
		}
		throw new TypeError(
			"Invalid output: output must be of type Array<{label:string; score:number; mask: string}>"
		);
	}

	function mouseout() {
		highlightIndex = -1;
	}

	function mouseover(index: number) {
		highlightIndex = index;
	}

	function mousemove(e: any, canvasW: number, canvasH: number) {
		let { layerX, layerY } = e;
		layerX = clip(layerX, 0, canvasW);
		layerY = clip(layerY, 0, canvasH);
		const row = Math.floor((layerX / canvasH) * imgH);
		const col = Math.floor((layerY / canvasW) * imgW);
		highlightIndex = -1;
		const index = (imgW * col + row) * 4;
		for (const [i, o] of output.entries()) {
			const pixel = o.imgData.data[index];
			if (pixel > 0) {
				highlightIndex = i;
			}
		}
	}

	function addOutputColor(output: ImageSegment[]) {
		output.forEach((val, index) => {
			const hash = mod(index, COLORS.length);
			const { color } = COLORS[hash];
			val.color = color;
		});
	}

	async function addOutputCanvasData(o_: ImageSegment): Promise<void> {
		const { mask, color } = o_;

		const colorToRgb = COLORS.reduce(
			(acc, cur) => ({ ...acc, [cur.color]: cur }),
			{}
		);
		const segmentImg = new Image();
		segmentImg.src = `data:image/png;base64, ${mask}`;
		// await image.onload
		await new Promise((resolve, _) => {
			segmentImg.onload = () => resolve(segmentImg);
		});
		imgW = segmentImg.naturalWidth;
		imgH = segmentImg.naturalHeight;

		const imgData = getImageData(segmentImg, imgW, imgH);
		const { r, g, b } = colorToRgb[color];
		const rgba = [r, g, b, maskOpacity];
		const background = Array(4).fill(0);

		for (let i = 0; i < imgData.data.length; i += 4) {
			setSlice(
				imgData.data,
				i,
				i + 4,
				imgData.data[i] === 255 ? rgba : background
			);
		}

		const bitmap = await createImageBitmap(imgData);

		o_.imgData = imgData;
		o_.bitmap = bitmap;
	}

	function getImageData(
		segmentImg: CanvasImageSource,
		imgW: number,
		imgH: number
	): ImageData {
		const tmpCanvas = document.createElement("canvas");
		tmpCanvas.width = imgW;
		tmpCanvas.height = imgH;
		const tmpCtx = tmpCanvas.getContext("2d");
		tmpCtx.drawImage(segmentImg, 0, 0, imgW, imgH);
		const segmentData = tmpCtx.getImageData(0, 0, imgW, imgH);
		return segmentData;
	}

	function setSlice(
		arr: Uint8ClampedArray,
		index_start: number,
		index_end: number,
		slice: Array<any>
	) {
		if (index_end - index_start !== slice.length) {
			throw new Error(
				`setSlice Error: lengths don't match ${index_end - index_start}!=${
					slice.length
				}`
			);
		}
		for (const [i, val] of slice.entries()) {
			arr[index_start + i] = val;
		}
	}

	// original: https://gist.github.com/MonsieurV/fb640c29084c171b4444184858a91bc7
	function polyfillCreateImageBitmap() {
		window.createImageBitmap = async function (
			data: ImageBitmapSource
		): Promise<ImageBitmap> {
			return new Promise((resolve, reject) => {
				let dataURL: string;
				if (data instanceof Blob) {
					dataURL = URL.createObjectURL(data);
				} else if (data instanceof ImageData) {
					const canvas = document.createElement("canvas");
					const ctx = canvas.getContext("2d");
					canvas.width = data.width;
					canvas.height = data.height;
					ctx.putImageData(data, 0, 0);
					dataURL = canvas.toDataURL();
				} else {
					reject(
						"createImageBitmap does not handle the provided image source type"
					);
				}
				const img = document.createElement("img");
				img.addEventListener("load", () => {
					resolve(img as any as ImageBitmap);
				});
				img.src = dataURL;
			});
		};
	}

	onMount(() => {
		if (typeof createImageBitmap === "undefined") {
			polyfillCreateImageBitmap();
		}
		getOutput(new Blob());
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
				classNames="no-hover:hidden"
				{isLoading}
				{imgSrc}
				{onSelectFile}
				onError={(e) => (error = e)}
			>
				{#if imgSrc}
					<Canvas {imgSrc} {highlightIndex} {mousemove} {mouseout} {output} />
				{/if}
			</WidgetDropzone>
			<!-- Better UX for mobile/table through CSS breakpoints -->
			{#if imgSrc}
				<Canvas
					classNames="mr-2 with-hover:hidden"
					{imgSrc}
					{highlightIndex}
					{mousemove}
					{mouseout}
					{output}
				/>
			{/if}
			<WidgetFileInput
				accept="image/*"
				classNames="mr-2 with-hover:hidden"
				{isLoading}
				label="Browse for image"
				{onSelectFile}
			/>
			{#if warning}
				<div class="alert alert-warning mt-2">{warning}</div>
			{/if}
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
