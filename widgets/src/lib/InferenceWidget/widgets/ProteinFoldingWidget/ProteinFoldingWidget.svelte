<script>
	import type { WidgetProps } from "../../shared/types";

	import WidgetProteinViewer from "../../shared/WidgetProteinViewer/WidgetProteinViewer.svelte";
	import WidgetFileInput from "../../shared/WidgetFileInput/WidgetFileInput.svelte";
	import WidgetSubmitBtn from "../../shared/WidgetSubmitBtn/WidgetSubmitBtn.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { getResponse } from "../../shared/helpers";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];

	let areSamplesVisible = true;
	let computeTime = "";
	let error: string = "";
	let file: Blob | File | null = null;
	let filename: string = "";
	let fileUrl: string;
	let isLoading = false;
	let isRecording = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output = "";
	let outputJson: string;
	let selectedSampleUrl = "";

	function onChangeRadio() {
		file = null;
		filename = "";
		fileUrl = "";
	}

	function onRecordStart() {
		areSamplesVisible = false;
		file = null;
		filename = "";
		fileUrl = "";
		isRecording = true;
	}

	function onSelectFile(updatedFile: Blob | File) {
		areSamplesVisible = false;
		isRecording = false;
		selectedSampleUrl = "";

		if (updatedFile.size !== 0) {
			const date = new Date();
			const time = date.toLocaleTimeString("en-US");
			filename =
				"name" in updatedFile
					? updatedFile.name
					: `Audio recorded from browser [${time}]`;
			file = updatedFile;
			fileUrl = URL.createObjectURL(file);
		}
	}

	async function getOutput(withModelLoading = false) {
		if (!file && !selectedSampleUrl) {
			error = "You must select or record an audio file";
			return;
		}

		const requestBody = file ? { file } : { url: selectedSampleUrl };

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
		output = "";
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

	function parseOutput(body: unknown): string {
		return body && typeof body === "object" && body instanceof Blob
			? URL.createObjectURL(body)
			: "";
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
			<div class="flex items-center flex-wrap">
				<WidgetFileInput accept="*/*" classNames="mt-1.5 mr-2" {onSelectFile} />
			</div>
			<WidgetSubmitBtn
				classNames="mt-2"
				isDisabled={isRecording}
				{isLoading}
				onClick={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetProteinViewer
			classNames="mt-4"
			src="http://localhost:8000/ranked_0.pdb"
		/>
		{#if output.length}
			<WidgetProteinViewer classNames="mt-4" src={output} />
		{/if}
	</svelte:fragment>
</WidgetWrapper>
