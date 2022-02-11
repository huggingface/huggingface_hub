<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetOutputText from "../../shared/WidgetOutputText/WidgetOutputText.svelte";
	import WidgetSubmitBtn from "../../shared/WidgetSubmitBtn/WidgetSubmitBtn.svelte";
	import WidgetTextarea from "../../shared/WidgetTextarea/WidgetTextarea.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import {
		addInferenceParameters,
		getDemoInputs,
		getResponse,
		getSearchParams,
		updateUrl,
	} from "../../shared/helpers";

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let callApiOnMount: WidgetProps["callApiOnMount"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];
	export let shouldUpdateUrl: WidgetProps["shouldUpdateUrl"];

	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output = "";
	let outputJson: string;
	let text = "";

	onMount(() => {
		const [textParam] = getSearchParams(["text"]);
		if (textParam) {
			text = textParam;
			getOutput();
		} else {
			const [demoText] = getDemoInputs(model, ["text"]);
			text = (demoText as string) ?? "";
			if (text && callApiOnMount) {
				getOutput();
			}
		}
	});

	async function getOutput(withModelLoading = false) {
		const trimmedValue = text.trim();

		if (!trimmedValue) {
			error = "You need to input some text";
			output = "";
			outputJson = "";
			return;
		}

		if (shouldUpdateUrl) {
			updateUrl({ text: trimmedValue });
		}

		const requestBody = { inputs: trimmedValue };
		addInferenceParameters(requestBody, model);

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
		if (Array.isArray(body) && body.length) {
			return body[0]?.["summary_text"] ?? "";
		}
		throw new TypeError(
			"Invalid output: output must be of type Array & non-empty"
		);
	}

	function previewInputSample(sample: Record<string, any>) {
		text = sample.text;
	}

	function applyInputSample(sample: Record<string, any>) {
		text = sample.text;
		getOutput();
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
		<form class="space-y-2">
			<WidgetTextarea bind:value={text} />
			<WidgetSubmitBtn
				{isLoading}
				onClick={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetOutputText classNames="mt-4" {output} />
	</svelte:fragment>
</WidgetWrapper>
