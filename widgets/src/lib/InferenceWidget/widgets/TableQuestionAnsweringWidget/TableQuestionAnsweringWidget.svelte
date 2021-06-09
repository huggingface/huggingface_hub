<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetQuickInput from "../../shared/WidgetQuickInput/WidgetQuickInput.svelte";
	import WidgetOutputTableQA from "../../shared/WidgetOutputTableQA/WidgetOutputTableQA.svelte";
	import WidgetTableInput from "../../shared/WidgetTableInput/WidgetTableInput.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import { parseJSON } from "../../../../../lib/ViewUtils";
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

	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Output | null = null;
	let outputJson: string;
	let table: string[][] = [[]];
	let query = "";

	onMount(() => {
		const [queryParam, tableParam] = getSearchParams(["query", "table"]);
		if (queryParam && tableParam) {
			query = queryParam;
			table = convertDataToTable((parseJSON(tableParam) as TableData) ?? {});
			getOutput();
		} else {
			const [demoQuery, demoTable] = getDemoInputs(model, ["text", "table"]);
			query = (demoQuery as string) ?? "";
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
		const trimmedQuery = query.trim();

		if (!trimmedQuery) {
			error = "You need to input a query";
			output = null;
			outputJson = "";
			return;
		}

		if (shouldUpdateUrl) {
			updateUrl({
				query: trimmedQuery,
				table: JSON.stringify(convertTableToData(table)),
			});
		}

		const requestBody = {
			inputs: {
				query: trimmedQuery,
				table: convertTableToData(table),
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
			<WidgetQuickInput
				bind:value={query}
				{isLoading}
				onClickSubmitBtn={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
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
	</svelte:fragment>
</WidgetWrapper>
