<script>
	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let classNames = "";
	export let checked: boolean;
	export let label: string;
</script>

<WidgetLabel classNames="inline-flex items-center {classNames}" {label}>
	<svelte:fragment slot="before">
		<input
			bind:checked
			class="rounded bg-gray-200 border-transparent focus:border-transparent text-blue-500 focus:ring-1 focus:ring-offset-2 focus:ring-gray-200 mr-2"
			type="checkbox"
		/>
	</svelte:fragment>
</WidgetLabel>
