<script>
	import type { WidgetProps } from "../../shared/types";
	import { mod } from "../../shared/ViewUtils";

	import WidgetCanvas from "./WidgetCanvas.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";
	import { onMount } from "svelte";
	import { highlightIndex as highlightIndexCanvas, updateCounter } from "./stores";

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
	}> = []; //TODO: define mask type
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

	$: mouseover($highlightIndexCanvas);
	$: outputWithColor = output.map((val, index) => {
		const hash = mod(index, COLORS.length);
		const color = COLORS[hash];
		return {...val, color};
	})

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

	function mouseout(): void {
		highlightIndex = -1;
		$highlightIndexCanvas = -1;
		$updateCounter++;
	}

	function mouseover(index: number): void {
		highlightIndex = index;
		$highlightIndexCanvas = index;
		$updateCounter++;
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
				{isLoading}
				bind:fileInput
				onChange={onSelectFile}
				{imgSrc}
				innerWidget={WidgetCanvas}
				innerWidgetProps={{ src: imgSrc, mouseover, mouseout, output:outputWithColor }}
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
