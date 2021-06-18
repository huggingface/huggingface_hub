<script>
	import type { WidgetProps } from "../../shared/types";

	import WidgetCheckbox from "../../shared/WidgetCheckbox/WidgetCheckbox.svelte";
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

	const accept = "image/*";
	let candidateLabels = "";
	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let multiClass = false;
	let output: Array<{ label: string; score: number }> = [];
	let outputJson: string;
	let imageBase64 = "";

	function onSelectFile(file: File) {
		error = file.type.match(accept) ? "" : "You need to upload an image";
		if (error) {
			imageBase64 = "";
			return;
		}

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
			inputs: imageBase64,
			parameters: {
				candidate_labels: trimmedCandidateLabels,
				multi_class: multiClass,
			},
		};

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
			getOutput(true);
		} else if (res.status === "error") {
			error = res.error;
		}
	}

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
		<form class="flex flex-col space-y-2">
			<WidgetDropzone {accept} {isLoading} {onSelectFile} />
			<WidgetTextInput
				bind:value={candidateLabels}
				label="Possible class names (comma-separated)"
				placeholder="Possible class names..."
			/>
			<WidgetCheckbox
				bind:checked={multiClass}
				label="Allow multiple true classes"
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
