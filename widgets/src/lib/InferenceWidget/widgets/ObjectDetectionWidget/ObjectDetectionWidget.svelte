<script>
	import type { WidgetProps, Box } from "../../shared/types";
	import { onMount } from "svelte";
	import { mod } from "../../shared/ViewUtils";

	import BoundingBoxes from "./SvgBoundingBoxes.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let imgSrc = "";
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Array<{
		label: string;
		score: number;
		box: Box;
	}> = [];
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
	$: outputWithColor = output.map((val, index) => {
		const hash = mod(index, COLORS.length);
		const color = COLORS[hash];
		return { ...val, color };
	});

	function onSelectFile(file: File | Blob) {
		getOutput(file);
	}

	async function getOutput(file: File | Blob, withModelLoading = false) {
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
	): arg is { label: string; score: number; box: Box }[] {
		return (
			Array.isArray(arg) &&
			arg.every(
				(x) =>
					typeof x.label === "string" &&
					typeof x.score === "number" &&
					x.box.xmin === "number" &&
					x.box.ymin === "number" &&
					x.box.xmax === "number" &&
					x.box.ymax === "number"
			)
		);
	}

	function parseOutput(
		body: unknown
	): Array<{ label: string; score: number; box: Box }> {
		if(isValidOutput(body)){
			return body;
		}
		throw new TypeError("Invalid output: output must be of type Array<{label:string; score:number; box:{xmin:number; ymin:number; xmax:number; ymax:number}}>");
	}

	function mouseout() {
		highlightIndex = -1;
	}

	function mouseover(index: number) {
		highlightIndex = index;
	}

	onMount(async () => {
		// imgSrc = "/cat.jpg";
		let objectDetectionData = await fetch("./od.json");
		output = await objectDetectionData.json();
		outputJson = JSON.stringify(output, null, 2);
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
				{onSelectFile}
				onError={(e) => (error = e)}
				bind:imgSrc
			>
				<BoundingBoxes
					{imgSrc}
					{mouseover}
					{mouseout}
					output={outputWithColor}
					{highlightIndex}
				/>
			</WidgetDropzone>
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
