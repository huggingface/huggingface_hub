<script lang="ts">
	import type { SvelteComponent } from "svelte";
	import DropdownEntry from "../DropdownEntry/DropdownEntry.svelte";
	import DropdownMenu from "../DropdownMenu/DropdownMenu.svelte";
	import IconCaretDown from "../Icons/IconCaretDown.svelte";

	export let classNames = "";
	export let btnClassNames = "";
	export let btnIcon: typeof SvelteComponent | undefined = undefined;
	export let btnIconClassNames = "";
	export let btnLabel = "";
	export let entries: Array<{
		classNames?: string;
		href?: string;
		icon?: typeof SvelteComponent;
		iconClassNames?: string;
		onClick?: (e: MouseEvent) => void;
		label?: string;
		noFollow?: boolean;
		underline?: boolean;
		targetBlank?: boolean;
	}> = [];
	export let forceMenuAlignement: "left" | "right" | undefined = undefined;
	export let menuClassNames = "";
	export let noBtnClass: boolean | undefined = undefined;
	export let withBtnCaret = false;

	let element: HTMLElement | undefined = undefined;
	let isOpen = false;
</script>

<div class="relative {classNames}" bind:this={element}>
	<!-- Button -->
	<button
		class="{btnClassNames}
			{!noBtnClass ? 'cursor-pointer w-full btn text-sm' : ''}"
		on:click={() => (isOpen = !isOpen)}
		type="button"
	>
		<!-- The "button" slot can overwrite the defaut button content -->
		{#if $$slots.button}
			<slot name="button" />
		{:else}
			{#if btnIcon}
				<svelte:component
					this={btnIcon}
					classNames="mr-1.5 {btnIconClassNames}"
				/>
			{/if}
			{btnLabel}
			{#if withBtnCaret}
				<IconCaretDown classNames="-mr-1 text-gray-500" />
			{/if}
		{/if}
	</button>
	<!-- /Button -->
	<!-- Menu -->
	{#if isOpen}
		<DropdownMenu
			classNames={menuClassNames}
			dropdownElement={element}
			forceAlignement={forceMenuAlignement}
			onClose={() => (isOpen = false)}
		>
			<!-- The "menu" slot can overwrite the defaut menu content -->
			{#if $$slots.menu}
				<slot name="menu" />
			{:else}
				{#each entries as entry}
					<DropdownEntry {...entry} />
				{/each}
			{/if}
		</DropdownMenu>
	{/if}
	<!-- Menu -->
</div>
