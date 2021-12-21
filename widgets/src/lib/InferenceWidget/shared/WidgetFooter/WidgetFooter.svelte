<script lang="ts">
	import Icon from "../../../Icon/Icon.svelte";
	import IconCode from "../../../Icon/data/IconCode";
	import IconMaximize from "../../../Icon/data/IconMaximize";

	export let onClickMaximizeBtn: () => void;
	export let outputJson: string;

	let isOutputJsonVisible = false;
</script>

<div class="mt-auto pt-4 flex items-center text-xs text-gray-500">
	<button
		class="flex items-center {outputJson
			? ''
			: 'cursor-not-allowed text-gray-300'}"
		disabled={!outputJson}
		on:click={() => {
			isOutputJsonVisible = !isOutputJsonVisible;
		}}
	>
		<Icon data={IconCode} classNames="mr-1" />
		JSON Output
	</button>
	<button
		class="flex items-center ml-auto"
		on:click|preventDefault={onClickMaximizeBtn}
	>
		<Icon data={IconMaximize} classNames="mr-1" />
		Maximize
	</button>
</div>
{#if outputJson && isOutputJsonVisible}
	<pre
		class="mt-3 max-h-screen overflow-auto text-xs text-gray-600 bg-gray-100 dark:bg-gray-800 p-2 rounded">{outputJson}</pre>
{/if}
