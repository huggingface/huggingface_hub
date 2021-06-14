<script>
	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let label: string = "";
	export let placeholder: string = "Your sentence here...";
	export let value: string;
</script>

<WidgetLabel {label}>
	<svelte:fragment slot="after">
		<textarea
			bind:value
			class="{label
				? 'mt-1.5'
				: ''} block w-full border border-gray-200 rounded-lg shadow-inner outline-none focus:ring-1 focus:ring-inset focus:ring-indigo-200 focus:shadow-inner dark:bg-gray-925"
			{placeholder}
			rows="3"
		/>
	</svelte:fragment>
</WidgetLabel>
