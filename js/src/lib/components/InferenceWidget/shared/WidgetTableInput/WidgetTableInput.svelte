<script lang="ts">
	import type { HighlightCoordinates } from "../types";

	import { onMount, tick } from "svelte";
	import { scrollToMax } from "../../../../utils/ViewUtils";
	import IconRow from "../../../Icons/IconRow.svelte";

	export let onChange: (table: (string | number)[][]) => void;
	export let highlighted: HighlightCoordinates;
	export let table: (string | number)[][] = [[]];
	export let canAddRow = true;
	export let canAddCol = true;

	let initialTable: (string | number)[][] = [[]];
	let tableContainerEl: HTMLElement;

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

	function editCell(e: Event, [x, y]) {
		const value = (e.target as HTMLElement)?.innerText;

		const updatedTable = table.map((row, rowIndex) =>
			rowIndex === y
				? row.map((col, colIndex) => (colIndex === x ? value : col))
				: row
		);
		onChange(updatedTable);
	}

	function onKeyDown(e: KeyboardEvent) {
		if (e.code == "Enter") {
			(e.target as HTMLElement)?.blur();
		}
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
						contenteditable={canAddCol}
						class="border-2 border-gray-100 h-6"
						on:keydown={onKeyDown}
						on:input={(e) => editCell(e, [x, 0])}
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
								" border-2 h-6"}
							contenteditable
							on:keydown={onKeyDown}
							on:input={(e) => editCell(e, [x, y + 1])}>{cell}</td
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
