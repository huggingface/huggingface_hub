<script>
	import type { WidgetProps } from "../../shared/types";

	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetImage from "../../shared/WidgetImage/WidgetImage.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";

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
	let output: Array<{ label: string; score: number }> = [];
	let outputJson: string;

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
			outputJson = res.outputJson;
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

	function isValidOutput(arg: any): arg is { label: string; score: number }[] {
		return (
			Array.isArray(arg) &&
			arg.every(
				(x) => typeof x.label === "string" && typeof x.score === "number"
			)
		);
	}

	function parseOutput(body: unknown): Array<{ label: string; score: number }> {
		return isValidOutput(body) ? body : [];
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
				innerWidget={WidgetImage}
				innerWidgetProps={{
					classNames: "pointer-events-none shadow mx-auto max-h-44",
					src: imgSrc,
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetOutputChart classNames="mt-4" {output} />
	</svelte:fragment>
</WidgetWrapper>
