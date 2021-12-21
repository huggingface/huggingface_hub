<script lang="ts">
	import Icon from "../../../Icon/Icon.svelte";
	import IconSpin from "../../../Icon/data/IconSpin";

	export let classNames = "";
	export let isDisabled = false;
	export let isLoading: boolean;
	export let label = "Compute";
	export let onClick: (e: MouseEvent) => void;
</script>

<button
	class="btn-widget w-24 h-10 px-5 {classNames}"
	disabled={isDisabled || isLoading}
	on:click|preventDefault={onClick}
	type="submit"
>
	{#if isLoading}
		<Icon data={IconSpin} classNames="text-gray-600 animate-spin" />
	{:else}
		{label}
	{/if}
</button>
