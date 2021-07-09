<script>
	import type { HighlightCoordinates } from "../types";

	import { onMount, tick } from "svelte";
	import { scrollToMax } from "../ViewUtils";
	import IconRow from "../../../Icons/IconRow.svelte";

	export let onChange: (table: string[][]) => void;
	export let highlighted: HighlightCoordinates = {};
	export let table: string[][] = [[]];
	export let canAddRow = true;
	export let canAddCol = true;

	let initialTable: string[][] = [[]];
	let tableContainerEl: HTMLElement = null;

	onMount(() => {
		initialTable = table.map((row) => row.map((cell) => cell));
	});

	async function addCol() {
		const updatedTable = table.map((row, colIndex) => [
			...row,
			colIndex === 0 ? `Header ${table[0].length + 1}` : "",
		]);
		onChange(updatedTable);
		await scrollTableToRight();
	}

	export async function scrollTableToRight() {
		await tick();
		scrollToMax(tableContainerEl, "x");
	}

	function addRow() {
		const updatedTable = [...table, Array(table[0].length).fill("")];
		onChange(updatedTable);
	}

	function editCell(e: KeyboardEvent, [x, y]) {
		const htmlElement = e.target as HTMLElement;
		const value = htmlElement?.innerText;
		if (e.code == "Enter") {
			htmlElement?.blur();
			return;
		}
		const updatedTable = table.map((row, rowIndex) =>
			rowIndex === y
				? row.map((col, colIndex) => (colIndex === x ? value : col))
				: row
		);
		onChange(updatedTable);
	}

	function resetTable() {
		const updatedTable = initialTable;
		onChange(updatedTable);
	}
</script>

<div class="overflow-auto" bind:this={tableContainerEl}>
	<table class="table-question-answering">
		<thead>
			<tr>
				{#each table[0] as header, x}
					<th
						contenteditable
						class="border-2 border-gray-100"
						on:keydown={(e) => editCell(e, [x, 0])}
					>
						{header}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each table.slice(1) as row, y}
				<tr class={highlighted[`${y}`] ?? "bg-white"}>
					{#each row as cell, x}
						<td
							class={(highlighted[`${y}-${x}`] ?? "border-gray-100") +
								" border-2"}
							contenteditable
							on:keydown={(e) => editCell(e, [x, y + 1])}>{cell}</td
						>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<div class="flex mb-1 flex-wrap">
	{#if canAddRow}
		<button
			class="btn-widget flex-1 lg:flex-none mt-2  mr-1.5"
			on:click={addRow}
			type="button"
		>
			<IconRow classNames="mr-2" />
			Add row
		</button>
	{/if}
	{#if canAddCol}
		<button
			class="btn-widget flex-1 lg:flex-none mt-2 lg:mr-1.5"
			on:click={addCol}
			type="button"
		>
			<IconRow classNames="transform rotate-90 mr-1" />
			Add col
		</button>
	{/if}
	<button
		class="btn-widget flex-1 mt-2 lg:flex-none lg:ml-auto"
		on:click={resetTable}
		type="button"
	>
		Reset table
	</button>
</div>
