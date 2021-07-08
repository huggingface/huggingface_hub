<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetOutputTableQA from "../../shared/WidgetOutputTableQA/WidgetOutputTableQA.svelte";
	import WidgetTableInput from "../../shared/WidgetTableInput/WidgetTableInput.svelte";
	import WidgetSubmitBtn from "../../shared/WidgetSubmitBtn/WidgetSubmitBtn.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { parseJSON } from "../../shared/ViewUtils";
	import {
		getDemoInputs,
		getResponse,
		getSearchParams,
		updateUrl,
	} from "../../shared/helpers";

	type TableData = Record<string, string[]>;

	interface Output {
		aggregator?: string;
		answer: string;
		coordinates: [number, number][];
		cells: number[];
	}

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let callApiOnMount: WidgetProps["callApiOnMount"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];
	export let shouldUpdateUrl: WidgetProps["shouldUpdateUrl"];

	const columns: string[] = Object.keys(
		model?.widgetData[0]?.structuredData ?? {}
	);

	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Output | null = null;
	let outputJson: string;
	let table: string[][] = [columns];
	let query = "How many stars does the transformers repository have?";

	onMount(() => {
		const [dataParam] = getSearchParams(["structuredData"]);
		if (dataParam) {
			table = convertDataToTable((parseJSON(dataParam) as TableData) ?? {});
			getOutput();
		} else {
			const [demoTable] = getDemoInputs(model, ["structuredData"]);
			table = convertDataToTable(demoTable as TableData);
			if (query && table && callApiOnMount) {
				getOutput();
			}
		}
	});

	function onChangeTable(updatedTable: string[][]) {
		table = updatedTable;
	}

	async function getOutput(withModelLoading = false) {
		if (table?.[0].length !== columns.length) {
			error = `Data needs to have ${columns.length} columns`;
			output = null;
			outputJson = "";
			return;
		}

		if (shouldUpdateUrl) {
			updateUrl({
				data: JSON.stringify(convertTableToData(table)),
			});
		}

		const requestBody = {
			inputs: {
				data: convertTableToData(table),
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
		output = null;
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

	function isValidOutput(arg: any): arg is Output {
		return (
			arg &&
			typeof arg === "object" &&
			typeof arg["answer"] === "string" &&
			Array.isArray(arg["coordinates"]) &&
			Array.isArray(arg["cells"])
		);
	}

	function parseOutput(body: unknown): Output | null {
		return isValidOutput(body) ? body : null;
	}

	/*
	 * Converts table from [[Header0, Header1, Header2], [Column0Val0, Column1Val0, Column2Val0], ...]
	 * to {Header0: [ColumnVal0, ...], Header1: [Column1Val0, ...], Header2: [Column2Val0, ...]}
	 */
	function convertTableToData(table: string[][]): TableData {
		return Object.fromEntries(
			table[0].map((cell, x) => {
				return [
					cell,
					table
						.slice(1)
						.flat()
						.filter((_, i) => i % table[0].length === x)
						.map((x) => String(x)), // some models can only handle strings (no numbers)
				];
			})
		);
	}

	/*
	 * Converts data from {Header0: [ColumnVal0, ...], Header1: [Column1Val0, ...], Header2: [Column2Val0, ...]}
	 * to [[Header0, Header1, Header2], [Column0Val0, Column1Val0, Column2Val0], ...]
	 */
	function convertDataToTable(data: TableData): string[][] {
		const dataArray = Object.entries(data); // [header, cell[]][]
		const nbCols = dataArray.length;
		const nbRows = (dataArray[0]?.[1]?.length ?? 0) + 1;
		return Array(nbRows)
			.fill("")
			.map((_, y) =>
				Array(nbCols)
					.fill("")
					.map((_, x) => (y === 0 ? dataArray[x][0] : dataArray[x][1][y - 1]))
			);
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
			<div class="mt-4">
				{#if output}
					<WidgetOutputTableQA {output} />
				{/if}
				{#if table.length > 1 || table[0].length > 1}
					<WidgetTableInput
						highlighted={output ? output.coordinates : []}
						onChange={onChangeTable}
						{table}
					/>
				{/if}
			</div>
			<WidgetSubmitBtn
				{isLoading}
				onClick={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom" />
</WidgetWrapper>
