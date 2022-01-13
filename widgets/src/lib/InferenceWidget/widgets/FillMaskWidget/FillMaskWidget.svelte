<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetOutputChart from "../../shared/WidgetOutputChart/WidgetOutputChart.svelte";
	import WidgetTextarea from "../../shared/WidgetTextarea/WidgetTextarea.svelte";
	import WidgetSubmitBtn from "../../shared/WidgetSubmitBtn/WidgetSubmitBtn.svelte";
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
	let output: Array<{ label: string; score: number }> = [];
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
		const trimmedText = text.trim();

		if (!trimmedText) {
			error = "You need to input some text";
			output = [];
			outputJson = "";
			return;
		}

		if (shouldUpdateUrl) {
			updateUrl({ text: trimmedText });
		}

		const requestBody = { inputs: trimmedText };
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
		if (Array.isArray(body)) {
			// entries = body -> text-classificartion
			// entries = body[0] -> summarization
			const entries = (
				model.pipeline_tag === "text-classification" ? body[0] ?? [] : body
			) as Record<string, unknown>[];
			return entries
				.filter((x) => !!x)
				.map((x) => ({
					// label = x.label -> text-classificartion
					label: x.label ? String(x.label) : String(x.token_str),
					score: x.score ? Number(x.score) : 0,
				}));
		}
		throw new TypeError("Invalid output: output must be of type Array");
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
		<form>
			<div class="text-sm text-gray-500 mb-1.5">
				Mask token: <code>{model.mask_token}</code>
			</div>
			<WidgetTextarea bind:value={text} />
			<WidgetSubmitBtn
				classNames="mt-2"
				{isLoading}
				onClick={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetOutputChart classNames="pt-4" {output} />
	</svelte:fragment>
</WidgetWrapper>
