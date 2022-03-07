<script lang="ts">
	import type { SvelteComponent } from "svelte";

	export let classNames = "";
	export let href: string | undefined = undefined;
	export let icon: typeof SvelteComponent | undefined = undefined;
	export let iconClassNames = "";
	export let label = "";
	export let noFollow = false;
	export let underline = false;
	export let onClick: (e: MouseEvent) => void = () => {};
	export let targetBlank = false;
</script>

<li>
	<a
		class="flex items-center hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer px-3 py-1.5 whitespace-nowrap 
			{classNames} {underline ? 'hover:underline' : ''}"
		{href}
		on:click={onClick}
		rel={noFollow ? "nofollow" : undefined}
		target={targetBlank ? "_blank" : undefined}
	>
		<!-- Adding children to the DropdownEntry element overwrite the default label/icon stuff -->
		{#if $$slots.default}
			<slot />
		{:else}
			{#if icon}
				<svelte:component this={icon} classNames="mr-1.5 {iconClassNames}" />
			{/if}
			{label}
		{/if}
	</a>
</li>
