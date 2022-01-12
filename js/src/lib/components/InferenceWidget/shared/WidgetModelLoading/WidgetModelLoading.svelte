<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import IconSpin from "../../../Icons/IconSpin.svelte";

	export let estimatedTime: number;

	let interval: any;
	let progressRatio = 0;
	let timeElapsed = 0;

	onMount(() => {
		interval = setInterval(() => {
			timeElapsed += 1;
			const ratio = timeElapsed / estimatedTime;
			progressRatio = ratio < 0.96 ? ratio : 0.96;
		}, 500);
	});

	onDestroy(() => {
		if (interval) {
			clearInterval(interval);
		}
	});
</script>

<div class="mt-3 flex h-10">
	<div
		class="z-0 flex flex-1 items-center justify-center rounded-lg bg-gray-50 relative text-gray-600 shadow-inner dark:bg-gray-950"
	>
		<div
			class="transition-all absolute inset-y-0 left-0 bg-gradient-to-r from-purple-200 to-purple-100 rounded-lg dark:from-purple-800 dark:to-purple-900"
			style="width: {progressRatio * 100}%;"
		/>
		<IconSpin
			classNames="text-purple-400 dark:text-purple-200 animate-spin mr-2 z-10"
		/>
		<span class="z-10">Model is loading</span>
	</div>
</div>
