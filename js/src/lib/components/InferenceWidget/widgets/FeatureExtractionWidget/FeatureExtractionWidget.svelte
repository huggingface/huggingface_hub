<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetQuickInput from "../../shared/WidgetQuickInput/WidgetQuickInput.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import {
		addInferenceParameters,
		getDemoInputs,
		getResponse,
		getSearchParams,
		updateUrl,
	} from "../../shared/helpers";
	import { DataTable } from "./DataTable";

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
	let output: DataTable | undefined;
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
			output = undefined;
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
		output = undefined;
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

	function parseOutput(body: any): DataTable {
		if (Array.isArray(body)) {
			if (body.length === 1) {
				body = body[0];
			}
			return new DataTable(body);
		}
		throw new TypeError("Invalid output: output must be of type Array");
	}

	const SINGLE_DIM_COLS = 4;

	function range(n: number, b?: number): number[] {
		return b
			? Array(b - n)
					.fill(0)
					.map((_, i) => n + i)
			: Array(n)
					.fill(0)
					.map((_, i) => i);
	}
	const numOfRows = (total_elems: number) => {
		return Math.ceil(total_elems / SINGLE_DIM_COLS);
	};

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
			<WidgetQuickInput
				bind:value={text}
				{isLoading}
				onClickSubmitBtn={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		{#if output}
			{#if output.isArrLevel0}
				<div class="mt-3 overflow-auto h-96">
					<table class="text-xs font-mono text-right border table-auto w-full">
						{#each range(numOfRows(output.oneDim.length)) as i}
							<tr>
								{#each range(SINGLE_DIM_COLS) as j}
									{#if j * numOfRows(output.oneDim.length) + i < output.oneDim.length}
										<td class="bg-gray-100 dark:bg-gray-900 text-gray-400 px-1">
											{j * numOfRows(output.oneDim.length) + i}
										</td>
										<td
											class="py-0.5 px-1 {output.bg(
												output.oneDim[j * numOfRows(output.oneDim.length) + i]
											)}"
										>
											{output.oneDim[
												j * numOfRows(output.oneDim.length) + i
											].toFixed(3)}
										</td>
									{/if}
								{/each}
							</tr>
						{/each}
					</table>
				</div>
			{:else}
				<div class="mt-3 overflow-auto">
					<table class="text-xs font-mono text-right border">
						<tr>
							<td class="bg-gray-100 dark:bg-gray-900" />
							{#each range(output.twoDim[0].length) as j}
								<td class="bg-gray-100 dark:bg-gray-900 text-gray-400 pt-1 px-1"
									>{j}</td
								>
							{/each}
						</tr>
						{#each output.twoDim as column, i}
							<tr>
								<td class="bg-gray-100 dark:bg-gray-900 text-gray-400 pl-4 pr-1"
									>{i}</td
								>
								{#each column as x}
									<td class="py-1 px-1 {output.bg(x)}">
										{x.toFixed(3)}
									</td>
								{/each}
							</tr>
						{/each}
					</table>
				</div>
			{/if}
		{/if}
	</svelte:fragment>
</WidgetWrapper>
