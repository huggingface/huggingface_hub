<script>
	import type { WidgetProps } from "../../shared/types";
	import { onMount } from "svelte";
	import { clip, mod, COLORS } from "../../shared/ViewUtils";
	import { getResponse } from "../../shared/helpers";

	import WidgetCanvas from "./WidgetCanvas.svelte";
	import WidgetFileInput from "../../shared/WidgetFileInput/WidgetFileInput.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import WidgetHeader from "../../shared/WidgetHeader/WidgetHeader.svelte";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	interface ImageSegments {
		png_string?: string;
		segments_info?: Array<{
			label: string;
			score: number;
			id: number;
		}>;
	}

	const maskOpacity = Math.floor(255 * 0.6);
	const idxAll = -1;

	let bitmaps: ImageBitmap[];
	let isCanvasAvailable = true;
	let computeTime = "";
	let error: string = "";
	let highlightIndex = idxAll;
	let idsFlat: number[] = [];
	let isLoading = false;
	let imgSrc = "";
	let imgW = 0;
	let imgH = 0;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: ImageSegments;
	let outputWithColor = [];
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
		outputWithColor = getOutputWithColor(output.segments_info);
		bitmaps = await getBitmaps(output.png_string);
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
		// 		outputWithColor = getOutputWithColor(output.segments_info);
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

	function isValidOutput(arg: any): arg is {
		png_string: string;
		segments_info: { label: string; score: number; id: number }[];
	} {
		return (
			typeof arg.png_string === "string" &&
			Array.isArray(arg.segments_info) &&
			arg.segments_info.every(
				(x) =>
					typeof x.label === "string" &&
					typeof x.score === "number" &&
					typeof x.id === "number"
			)
		);
	}
	function parseOutput(body: unknown): ImageSegments {
		if (isValidOutput(body)) {
			return body;
		}
		throw new TypeError(
			"Invalid output: output must be of type <png_string: string; segments_info: Array<{label:string; score:number; id:number}>>"
		);
	}

	function mouseout() {
		highlightIndex = idxAll;
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
		highlightIndex = idsFlat[imgW * col + row];
	}

	function getOutputWithColor(
		segments_info: Array<{
			label: string;
			score: number;
			id: number;
		}>
	): Array<{
		label: string;
		score: number;
		id: number;
		color: string;
	}> {
		return segments_info.map((val, index) => {
			const hash = mod(index, COLORS.length);
			const { color } = COLORS[hash];
			return { ...val, color };
		});
	}

	async function getBitmaps(png_string: string): Promise<ImageBitmap[]> {
		const idToColor = outputWithColor.reduce(
			(acc, cur) => ({ ...acc, [cur.id]: cur.color }),
			{}
		);
		const colorToRgb = COLORS.reduce(
			(acc, cur) => ({ ...acc, [cur.color]: cur }),
			{}
		);
		const segmentImg = new Image();
		segmentImg.src = `data:image/png;base64, ${png_string}`;
		// await image.onload
		await new Promise((resolve, _) => {
			segmentImg.onload = () => resolve(segmentImg);
		});
		imgW = segmentImg.naturalWidth;
		imgH = segmentImg.naturalHeight;

		idsFlat = [];
		const { segmentData, imagesData } = getImagesData(segmentImg, imgW, imgH);

		for (let i = 0; i < segmentData.data.length; i += 4) {
			const [r, g, b] = segmentData.data.slice(i, i + 3);
			const id = r + 256 * g + 256 * 256 * b;
			idsFlat.push(id);
			const color = idToColor[id];
			if (color) {
				const { r, g, b } = colorToRgb[color];
				const rgba = [r, g, b, maskOpacity];
				setSlice(imagesData[idxAll].data, i, i + 4, rgba);
				setSlice(imagesData[id].data, i, i + 4, rgba);
			}
		}

		const bitmaps: ImageBitmap[] = [];
		for (const key in imagesData) {
			bitmaps[key] = await createImageBitmap(imagesData[key]);
		}

		return bitmaps;
	}

	function getImagesData(
		segmentImg: CanvasImageSource,
		imgW: number,
		imgH: number
	): {
		segmentData: ImageData;
		imagesData: {
			[key: number]: ImageData;
		};
	} {
		const tmpCanvas = document.createElement("canvas");
		tmpCanvas.width = imgW;
		tmpCanvas.height = imgH;
		const tmpCtx = tmpCanvas.getContext("2d");
		tmpCtx.drawImage(segmentImg, 0, 0, imgW, imgH);
		const segmentData = tmpCtx.getImageData(0, 0, imgW, imgH);
		const imagesData = {};
		imagesData[idxAll] = tmpCtx.createImageData(imgW, imgH);
		for (const val of outputWithColor) {
			imagesData[val.id] = tmpCtx.createImageData(imgW, imgH);
		}
		return { segmentData, imagesData };
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

	onMount(() => {
		if (typeof createImageBitmap === "undefined") {
			isCanvasAvailable = false;
		}
		// getOutput(new Blob());
	});
</script>

{#if isCanvasAvailable}
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
						<WidgetCanvas
							{imgSrc}
							{bitmaps}
							{highlightIndex}
							{mousemove}
							{mouseout}
						/>
					{/if}
				</WidgetDropzone>
				<!-- Better UX for mobile/table through CSS breakpoints -->
				{#if imgSrc}
					<WidgetCanvas
						classNames="mr-2 with-hover:hidden"
						{imgSrc}
						{bitmaps}
						{highlightIndex}
						{mousemove}
						{mouseout}
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
				output={outputWithColor}
				{highlightIndex}
				{mouseover}
				{mouseout}
			/>
		</svelte:fragment>
	</WidgetWrapper>
{:else}
	<WidgetHeader noTitle={false} pipeline={model.pipeline_tag} />
	<p class="text-gray-500 text-sm">
		This widget is supported on Chrome, FireFox, & Edge
	</p>
{/if}
