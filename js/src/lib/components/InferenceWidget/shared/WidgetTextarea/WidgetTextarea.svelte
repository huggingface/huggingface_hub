<script>
	import { tick } from "svelte";

	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let label: string = "";
	export let placeholder: string = "Your sentence here...";
	export let value: string;

	let textAreaEl: HTMLTextAreaElement;

	const HEIGHT_LIMIT = 500 as const;

	async function resize() {
		if (!!textAreaEl) {
			textAreaEl.style.height = "0px";
			await tick();
			textAreaEl.style.height =
				Math.min(textAreaEl.scrollHeight, HEIGHT_LIMIT) + "px";
		}
	}

	$: {
		value;
		resize();
	}
</script>

<WidgetLabel {label}>
	<svelte:fragment slot="after">
		<textarea
			bind:this={textAreaEl}
			bind:value
			class="{label
				? 'mt-1.5'
				: ''} block w-full border border-gray-200 rounded-lg shadow-inner outline-none focus:ring-1 focus:ring-inset focus:ring-indigo-200 focus:shadow-inner dark:bg-gray-925"
			{placeholder}
		/>
	</svelte:fragment>
</WidgetLabel>
