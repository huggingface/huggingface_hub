<script>
	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let classNames = "";
	export let group: string;
	export let label: string;
	export let onChange: (e: Event) => void = () => null;
	export let value: string;
</script>

<WidgetLabel classNames="inline-flex items-center {classNames}" {label}>
	<svelte:fragment slot="before">
		<input
			bind:group
			class="bg-gray-200 dark:bg-gray-700 shadow-inner border-transparent focus:border-transparent text-blue-500 ring-1 ring-offset-2 ring-gray-200 dark:ring-gray-700 mr-2"
			on:change={onChange}
			type="radio"
			{value}
		/>
	</svelte:fragment>
</WidgetLabel>
