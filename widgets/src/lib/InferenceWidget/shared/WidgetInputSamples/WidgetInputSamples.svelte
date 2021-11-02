<script>
	import { slide } from "svelte/transition";

	export let inputSamples: Record<string, any>[];
	export let applyInputSample: (sample: Record<string, any>) => void;
	export let previewInputSample: (sample: Record<string, any>) => void;

	let isOptionsVisible = false;
	let isTouchOptionClicked = false;
	let selectedIdx: number;
	let title = "Examples";

	function _applyInputSample(idx: number) {
		isOptionsVisible = false;
		isTouchOptionClicked = false;
		const sample = inputSamples[idx];
		title = sample.example_title;
		applyInputSample(sample);
	}

	function _previewInputSample(idx: number, isTocuh = false) {
		const sample = inputSamples[idx];
		previewInputSample(sample);
		if(isTocuh){
			isTouchOptionClicked = true;
			selectedIdx = idx;
		}
	}

	function toggleOptionsVisibility() {
		isOptionsVisible = !isOptionsVisible;
	}
</script>

<div class="relative z-10">
	<div
		class="no-hover:hidden inline-flex justify-between w-32 lg:w-44 rounded-md border border-gray-100 px-4 py-1"
		on:click={toggleOptionsVisibility}
	>
		<p class="text-sm truncate">{title}</p>
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
			<p class="text-sm truncate">{title}</p>
			<svg
				class="-mr-1 ml-2 h-5 w-5 transition ease-in-out transform"
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
		<div
		class="with-hover:hidden inline-flex justify-center w-32 lg:w-44 rounded-md border border-green-500 px-4 py-1"
		on:click={() => _applyInputSample(selectedIdx)}
	>
			 <p class="text-green-500">Confirm</p>
			</div>
		{/if}

	{#if isOptionsVisible}
		<div
			class="origin-top-right absolute right-0 mt-1 w-full rounded-md ring-1 ring-black ring-opacity-10"
			transition:slide
		>
			<div class="py-1 bg-white rounded-md" role="none">
				{#each inputSamples as { example_title }, i}
					<p
						class="no-hover:hidden px-4 py-2 text-sm hover:bg-gray-100 hover:text-gray-900 dark:hover:bg-gray-800 dark:hover:text-gray-200"
						on:mouseover={() => _previewInputSample(i)}
						on:click={() => _applyInputSample(i)}
					>
						{example_title}
					</p>
					<p
						class="with-hover:hidden px-4 py-2 text-sm hover:bg-gray-100 hover:text-gray-900 dark:hover:bg-gray-800 dark:hover:text-gray-200"
						on:click={() => _previewInputSample(i, true)}
					>
						{example_title}
					</p>
				{/each}
			</div>
		</div>
	{/if}
</div>
