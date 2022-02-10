<script>
	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let label: string | undefined = undefined;
	export let placeholder: string = "Your sentence here...";
	export let value: string;
</script>

<WidgetLabel {label}>
	<svelte:fragment slot="after">
		<input
			bind:value
			class="{label ? 'mt-1.5' : ''} form-input-alt block w-full "
			{placeholder}
			type="text"
		/>
	</svelte:fragment>
</WidgetLabel>
