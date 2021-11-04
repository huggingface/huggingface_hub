<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetOutputConvo from "../../shared/WidgetOutputConvo/WidgetOutputConvo.svelte";
	import WidgetQuickInput from "../../shared/WidgetQuickInput/WidgetQuickInput.svelte";
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

	interface Conversation {
		generated_responses: string[];
		past_user_inputs: string[];
	}
	interface Response {
		conversation: Conversation;
		generated_text: string;
	}

	type Output = Array<{
		input: string;
		response: string;
	}>;

	let computeTime = "";
	let conversation: {
		generated_responses: string[];
		past_user_inputs: string[];
	} = {
		generated_responses: [],
		past_user_inputs: [],
	};
	let error: string = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Output = [];
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
			return;
		}

		if (shouldUpdateUrl && !conversation.past_user_inputs.length) {
			updateUrl({ text: trimmedText });
		}

		const requestBody = {
			inputs: {
				generated_responses: conversation.generated_responses,
				past_user_inputs: conversation.past_user_inputs,
				text: trimmedText,
			},
		};
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
		outputJson = "";

		if (res.status === "success") {
			computeTime = res.computeTime;
			outputJson = res.outputJson;
			if (res.output) {
				conversation = res.output.conversation;
				output = res.output.output;
			}
			// Emptying input value
			text = "";
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

	function isValidOutput(arg: any): arg is Response {
		return (
			arg &&
			Array.isArray(arg?.conversation?.generated_responses) &&
			Array.isArray(arg?.conversation?.past_user_inputs)
		);
	}

	function parseOutput(body: unknown): {
		conversation: Conversation;
		output: Output;
	} {
		if (isValidOutput(body)) {
			const conversation = body.conversation;
			const pastUserInputs = conversation.past_user_inputs;
			const generatedResponses = conversation.generated_responses;
			const output = pastUserInputs
				.filter((x, i) => x != null && generatedResponses[i] != null) // != null -> not null OR undefined
				.map((x, i) => ({
					input: x ?? "",
					response: generatedResponses[i] ?? "",
				}));
			return { conversation, output };
		}
		throw new TypeError(
			"Invalid output: output must be of type <conversation: <generated_responses:Array; past_user_inputs:Array>>"
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
		<WidgetOutputConvo modelId={model.id} {output} />
		<form>
			<WidgetQuickInput
				bind:value={text}
				flatTop={true}
				{isLoading}
				onClickSubmitBtn={() => {
					getOutput();
				}}
				submitButtonLabel="Send"
			/>
		</form>
	</svelte:fragment>
</WidgetWrapper>
