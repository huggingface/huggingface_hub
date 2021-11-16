<script>
	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let label: string = "";
	export let placeholder: string = "Your sentence here...";
	export let value: string;
</script>

<WidgetLabel {label}>
	<svelte:fragment slot="after">
		<span
			bind:textContent={value}
			class="{label
				? 'mt-1.5'
				: ''} block overflow-auto resize-y py-1 px-2 w-full max-h-[500px] border border-gray-200 rounded-lg shadow-inner outline-none focus:ring-1 focus:ring-inset focus:ring-indigo-200 focus:shadow-inner dark:bg-gray-925"
			role="textbox"
			contenteditable
			style="--placeholder: '{placeholder}'"
		/>
	</svelte:fragment>
</WidgetLabel>

<style>
	span[contenteditable]:empty::before {
		content: var(--placeholder);
		color: rgba(156, 163, 175);
	}
</style>
