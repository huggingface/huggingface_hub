<script lang="ts">
	import { slide } from "svelte/transition";

	export let isLoading = false;
	export let inputSamples: Record<string, any>[];
	export let applyInputSample: (sample: Record<string, any>) => void;
	export let previewInputSample: (sample: Record<string, any>) => void;

	let containerEl: HTMLElement;
	let isOptionsVisible = false;
	let isTouchOptionClicked = false;
	let touchSelectedIdx: number;
	let title = "Examples";

	function _applyInputSample(idx: number) {
		hideOptions();
		const sample = inputSamples[idx];
		title = sample.example_title;
		applyInputSample(sample);
	}

	function _previewInputSample(idx: number, isTocuh = false) {
		const sample = inputSamples[idx];
		if (isTocuh) {
			isTouchOptionClicked = true;
			touchSelectedIdx = idx;
		}
		previewInputSample(sample);
	}

	function toggleOptionsVisibility() {
		isOptionsVisible = !isOptionsVisible;
	}

	function onClick(e: MouseEvent | TouchEvent) {
		let targetElement = e.target;
		do {
			if (targetElement == containerEl) {
				// This is a click inside. Do nothing, just return.
				return;
			}
			targetElement = (targetElement as HTMLElement).parentElement;
		} while (targetElement);
		// This is a click outside
		hideOptions();
	}

	function hideOptions() {
		isOptionsVisible = false;
		isTouchOptionClicked = false;
	}
</script>

<svelte:window on:click={onClick} />

<div
	class="relative mb-1.5
		{isLoading && 'pointer-events-none opacity-50'} 
		{isOptionsVisible && 'z-10'}"
	bind:this={containerEl}
>
	<div
		class="no-hover:hidden inline-flex justify-between w-32 lg:w-44 rounded-md border border-gray-100 px-4 py-1"
		on:click={toggleOptionsVisibility}
	>
		<div class="text-sm truncate">{title}</div>
		<svg
			class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform {isOptionsVisible &&
				'-rotate-180'}"
			xmlns="http://www.w3.org/2000/svg"
			viewBox="0 0 20 20"
			fill="currentColor"
			aria-hidden="true"
		>
			<path
				fill-rule="evenodd"
				d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
				clip-rule="evenodd"
			/>
		</svg>
	</div>
	{#if !isTouchOptionClicked}
		<div
			class="with-hover:hidden inline-flex justify-between w-32 lg:w-44 rounded-md border border-gray-100 px-4 py-1"
			on:click={toggleOptionsVisibility}
		>
			<div class="text-sm truncate">{title}</div>
			<svg
				class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform {isOptionsVisible &&
					'-rotate-180'}"
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 20 20"
				fill="currentColor"
				aria-hidden="true"
			>
				<path
					fill-rule="evenodd"
					d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
					clip-rule="evenodd"
				/>
			</svg>
		</div>
	{:else}
		<!-- Better UX for mobile/table through CSS breakpoints -->
		<div
			class="with-hover:hidden inline-flex justify-center w-32 lg:w-44 rounded-md border border-green-500 px-4 py-1"
			on:click={() => _applyInputSample(touchSelectedIdx)}
		>
			<div class="text-green-500">Compute</div>
		</div>
	{/if}

	{#if isOptionsVisible}
		<div
			class="origin-top-right absolute right-0 mt-1 w-full rounded-md ring-1 ring-black ring-opacity-10"
			transition:slide
		>
			<div class="py-1 bg-white rounded-md" role="none">
				{#each inputSamples as { example_title }, i}
					<div
						class="no-hover:hidden px-4 py-2 text-sm hover:bg-gray-100 hover:text-gray-900 dark:hover:bg-gray-800 dark:hover:text-gray-200"
						on:mouseover={() => _previewInputSample(i)}
						on:click={() => _applyInputSample(i)}
					>
						{example_title}
					</div>
					<!-- Better UX for mobile/table through CSS breakpoints -->
					<div
						class="with-hover:hidden px-4 py-2 text-sm hover:bg-gray-100 hover:text-gray-900 dark:hover:bg-gray-800 dark:hover:text-gray-200"
						on:click={() => _previewInputSample(i, true)}
					>
						{example_title}
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>
