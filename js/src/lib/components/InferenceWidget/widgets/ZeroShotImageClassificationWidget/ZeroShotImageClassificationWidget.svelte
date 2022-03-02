<script lang="ts">
	import type { WidgetProps } from "../../shared/types";

	import WidgetFileInput from "../../shared/WidgetFileInput/WidgetFileInput.svelte";
	import WidgetDropzone from "../../shared/WidgetDropzone/WidgetDropzone.svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetSubmitBtn from "../../shared/WidgetSubmitBtn/WidgetSubmitBtn.svelte";
	import WidgetTextInput from "../../shared/WidgetTextInput/WidgetTextInput.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	let candidateLabels = "";
	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let imgSrc = "";
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Array<{ label: string; score: number }> = [];
	let outputJson: string;
	let imageBase64 = "";

	function onSelectFile(file: File | Blob) {
		imgSrc = URL.createObjectURL(file);

		let fileReader: FileReader = new FileReader();
		fileReader.onload = () => {
			const imageBase64WithPrefix: string = fileReader.result as string;
			imageBase64 = imageBase64WithPrefix.split(",")[1]; // remove prefix
			isLoading = false;
		};
		isLoading = true;
		fileReader.readAsDataURL(file);
	}

	async function getOutput(withModelLoading = false) {
		const trimmedCandidateLabels = candidateLabels.trim().split(",").join(",");

		if (!imageBase64) {
			error = "You need to upload an image";
			output = [];
			outputJson = "";
			return;
		}

		if (!trimmedCandidateLabels) {
			error = "You need to input at least one label";
			output = [];
			outputJson = "";
			return;
		}

		const requestBody = {
			inputs: {
				image: imageBase64,
				candidate_labels: trimmedCandidateLabels,
			},
		};

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
			getOutput(true);
		} else if (res.status === "error") {
			error = res.error;
		}
	}

	// TODO
	function parseOutput(body: unknown): Array<{ label: string; score: number }> {
		if (
			body &&
			typeof body === "object" &&
			Array.isArray(body["labels"]) &&
			Array.isArray(body["scores"])
		) {
			return body["labels"]
				.filter((_, i) => body["scores"][i] != null) // != null -> not null OR undefined
				.map((x, i) => ({
					label: x ?? "",
					score: body["scores"][i] ?? 0,
				}));
		}
		return [];
	}

	async function applyInputSample(sample: Record<string, any>) {
		// imgSrc = sample.src;
		// const blob = await getBlobFromUrl(imgSrc);
		// getOutput(blob);
	}

	function previewInputSample(sample: Record<string, any>) {
		// imgSrc = sample.src;
		// output = [];
		// outputJson = "";
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
		<form class="flex flex-col space-y-2">
			<WidgetDropzone
				classNames="no-hover:hidden"
				{isLoading}
				{imgSrc}
				{onSelectFile}
				onError={(e) => (error = e)}
			>
				{#if imgSrc}
					<img
						src={imgSrc}
						class="pointer-events-none shadow mx-auto max-h-44"
						alt=""
					/>
				{/if}
			</WidgetDropzone>
			<!-- Better UX for mobile/table through CSS breakpoints -->
			{#if imgSrc}
				<div
					class="mb-2 flex justify-center bg-gray-50 dark:bg-gray-900 with-hover:hidden"
				>
					<img src={imgSrc} class="pointer-events-none max-h-44" alt="" />
				</div>
			{/if}
			<WidgetFileInput
				accept="image/*"
				classNames="mr-2 with-hover:hidden"
				{isLoading}
				label="Browse for image"
				{onSelectFile}
			/>
			<WidgetTextInput
				bind:value={candidateLabels}
				label="Possible class names (comma-separated)"
				placeholder="Possible class names..."
			/>
			<WidgetSubmitBtn
				{isLoading}
				onClick={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		{#if output.length}
			<WidgetOutputChart classNames="mt-4" {output} />
		{/if}
	</svelte:fragment>
</WidgetWrapper>
