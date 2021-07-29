<script>
	import { onMount } from "svelte";
	import type { WidgetProps } from "../../shared/types";
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
		box: any;
	}> = []; //TODO: define box type
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
	): arg is { label: string; score: number; box: any }[] {
		return (
			Array.isArray(arg) &&
			arg.every(
				(x) => typeof x.label === "string" && typeof x.score === "number"
				// TODO: check box type
			)
		);
	}

	function parseOutput(
		body: unknown
	): Array<{ label: string; score: number; box: any }> {
		return isValidOutput(body) ? body : [];
	}

	function mouseout(): void {
		highlightIndex = -1;
	}

	function mouseover(index: number): void {
		highlightIndex = index;
	}

	onMount(async () => {
		imgSrc = "/cat.jpg";
		let objectDetectionData = await fetch("./od.json");
		output = await objectDetectionData.json();
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
				{imgSrc}
				innerWidget={BoundingBoxes}
				innerWidgetProps={{
					src: imgSrc,
					mouseover,
					mouseout,
					output: outputWithColor,
					highlightIndex,
				}}
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
