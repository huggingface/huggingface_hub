<script>
	import type { WidgetProps } from "../../shared/types";
	import { mod } from "../../shared/ViewUtils";

	import WidgetCanvas from "./WidgetCanvas.svelte";
	import WidgetFileInput from "../../shared/WidgetFileInput/WidgetFileInput.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";
	import { onMount } from "svelte";
	// import {
	// 	highlightIndex as highlightIndexCanvas,
	// 	updateCounter,
	// } from "./stores";

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

	let computeTime = "";
	let error: string = "";
	let fileInput: HTMLInputElement;
	let isLoading = false;
	let imgSrc = "";
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: ImageSegments;
	let outputJson: string;
	let highlightIndex = -1;

	const COLORS = [
		"red",
		"green",
		"yellow",
		"blue",
		"orange",
		"purple",
		"cyan",
		"lime",
	] as const;

	// $: mouseover($highlightIndexCanvas);
	$: outputWithColor = output?.segments_info.map((val, index) => {
		const hash = mod(index, COLORS.length);
		const color = COLORS[hash];
		return { ...val, color };
	});

	function onSelectFile(file: File | Blob) {
		imgSrc = URL.createObjectURL(file);
		getOutput(file);
	}

	async function getOutput(file: File | Blob, withModelLoading = false) {
		if (!file) {
			return;
		}

		// Reset values
		computeTime = "";
		error = "";
		output = {};
		outputJson = "";

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
		modelLoading = { isLoading: false, estimatedTime: 0 };

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
					typeof x.score === "number"
			)
		);
	}

	function parseOutput(body: unknown): ImageSegments {
		if (isValidOutput(body)) {
			return body;
		}
		throw new TypeError(
			"Invalid output: output must be of type Array<{label:string; score:number; box:{xmin:number; ymin:number; xmax:number; ymax:number}}>"
		);
	}

	function mouseout(): void {
		highlightIndex = -1;
		// $highlightIndexCanvas = -1;
		// $updateCounter++;
	}

	function mouseover(index: number): void {
		highlightIndex = index;
		// $highlightIndexCanvas = index;
		// $updateCounter++;
	}
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
					<WidgetCanvas
						{imgSrc}
						{mouseover}
						{mouseout}
						output={outputWithColor}
						{highlightIndex}
					/>
				{/if}
			</WidgetDropzone>
			<!-- Better UX for mobile/table through CSS breakpoints -->
			{#if imgSrc}
				<WidgetCanvas
					classNames="mb-2 with-hover:hidden"
					{imgSrc}
					{mouseover}
					{mouseout}
					output={outputWithColor}
					{highlightIndex}
				/>
			{/if}
			<WidgetFileInput
				accept="image/*"
				classNames="mr-2 with-hover:hidden"
				{isLoading}
				label="Browse for image"
				{onSelectFile}
			/>
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
