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
			class="bg-gray-200 border-transparent focus:border-transparent text-blue-500 focus:ring-1 focus:ring-offset-2 focus:ring-gray-200 mr-2"
			on:change={onChange}
			type="radio"
			{value}
		/>
	</svelte:fragment>
</WidgetLabel>
