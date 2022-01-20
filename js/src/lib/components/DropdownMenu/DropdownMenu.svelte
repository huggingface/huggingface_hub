<script lang="ts">
	import { onMount } from "svelte";

	type Alignement = "left" | "right";

	export let classNames = "";
	export let dropdownElement: HTMLElement | undefined = undefined;
	export let forceAlignement: Alignement | undefined = undefined;
	export let onClose: () => void;

	// MUST be set to left if forceAlignement is undefined or else
	// the browser won't be able to properly compute x and width
	let alignement: Alignement = forceAlignement ?? "left";
	let element: HTMLElement | undefined;

	onMount(() => {
		document.addEventListener("click", handleClickDocument);

		if (!forceAlignement) {
			const docWidth = document.documentElement.clientWidth;
			const domRect = element?.getBoundingClientRect() || {};
			const left = domRect["left"] ?? 0;
			const width = domRect["width"] ?? 0;
			alignement = left + width > docWidth ? "right" : "left";
		}

		return () => {
			document.removeEventListener("click", handleClickDocument);
		};
	});

	function handleClickDocument(e: MouseEvent) {
		// We ignore clicks that happens inside the Dropdown itself
		// (prevent race condition  with other click handlers)
		const targetElement = e.target as HTMLElement;
		if (
			targetElement !== dropdownElement &&
			!dropdownElement?.contains(targetElement)
		) {
			onClose();
		}
	}
</script>

<div
	bind:this={element}
	class="absolute top-full mt-1 min-w-full w-auto bg-white rounded-xl overflow-hidden shadow-lg z-10 border border-gray-100
		{alignement === 'right' ? 'right-0' : 'left-0'}
		{classNames}"
	on:click={onClose}
>
	<ul class="min-w-full w-auto">
		<slot />
	</ul>
</div>
