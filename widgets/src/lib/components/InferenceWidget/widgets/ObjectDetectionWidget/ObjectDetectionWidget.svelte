<script>
	import type { WidgetProps, DetectedObject } from "../../shared/types";
	import { COLORS } from "../../shared/consts";
	import { mod } from "../../../../utils/ViewUtils";

	import BoundingBoxes from "./SvgBoundingBoxes.svelte";
	import WidgetFileInput from "../../shared/WidgetFileInput/WidgetFileInput.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse, getBlobFromUrl } from "../../shared/helpers";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	let computeTime = "";
	let error: string = "";
	let warning: string = "";
	let isLoading = false;
	let imgSrc = "";
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: DetectedObject[] = [];
	let outputJson: string;
	let highlightIndex = -1;

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
		warning = "";
		output = [];
		outputJson = "";

		const requestBody = { file };

		isLoading = true;

		const res = await getResponse(
			apiUrl,
			model.id,
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
			output = output.map((o, idx) => addOutputColor(o, idx));
			outputJson = res.outputJson;
			if (output.length === 0) {
				warning = "No object was detected";
			}
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

	function addOutputColor(detObj: DetectedObject, idx: number) {
		const hash = mod(idx, COLORS.length);
		const { color } = COLORS[hash];
		return { ...detObj, color };
	}

	function isValidOutput(arg: any): arg is DetectedObject[] {
		return (
			Array.isArray(arg) &&
			arg.every(
				(x) =>
					typeof x.label === "string" &&
					typeof x.score === "number" &&
					typeof x.box.xmin === "number" &&
					typeof x.box.ymin === "number" &&
					typeof x.box.xmax === "number" &&
					typeof x.box.ymax === "number"
			)
		);
	}

	function parseOutput(body: unknown): DetectedObject[] {
		if (isValidOutput(body)) {
			return body;
		}
		throw new TypeError(
			"Invalid output: output must be of type Array<{label:string; score:number; box:{xmin:number; ymin:number; xmax:number; ymax:number}}>"
		);
	}

	function mouseout() {
		highlightIndex = -1;
	}

	function mouseover(index: number) {
		highlightIndex = index;
	}

	async function applyInputSample(sample: Record<string, any>) {
		imgSrc = sample.src;
		const blob = await getBlobFromUrl(imgSrc);
		getOutput(blob);
	}

	function previewInputSample(sample: Record<string, any>) {
		imgSrc = sample.src;
		output = [];
		outputJson = "";
	}
</script>

<WidgetWrapper
	{apiUrl}
	{applyInputSample}
	{computeTime}
	{error}
	{isLoading}
	{model}
	{modelLoading}
	{noTitle}
	{outputJson}
	{previewInputSample}
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
					<BoundingBoxes
						{imgSrc}
						{mouseover}
						{mouseout}
						{output}
						{highlightIndex}
					/>
				{/if}
			</WidgetDropzone>
			<!-- Better UX for mobile/table through CSS breakpoints -->
			{#if imgSrc}
				<BoundingBoxes
					classNames="mb-2 with-hover:hidden"
					{imgSrc}
					{mouseover}
					{mouseout}
					{output}
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
			{#if warning}
				<div class="alert alert-warning mt-2">{warning}</div>
			{/if}
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetOutputChart
			classNames="pt-4"
			{output}
			{highlightIndex}
			{mouseover}
			{mouseout}
		/>
	</svelte:fragment>
</WidgetWrapper>
