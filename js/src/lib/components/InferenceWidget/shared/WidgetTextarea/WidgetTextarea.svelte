<script>
	import WidgetLabel from "../WidgetLabel/WidgetLabel.svelte";

	export let label: string = "";
	export let placeholder: string = "Your sentence here...";
	export let value: string;

	// hack to handle FireFox contenteditable bug
	let innterHTML: string;
	let spanEl: HTMLSpanElement;
	const REGEX_SPAN = /<span .+>(.*)<\/span>/gms;
	$: {
		if (spanEl && innterHTML && REGEX_SPAN.test(innterHTML)) {
			innterHTML = innterHTML.replace(REGEX_SPAN, (_, txt) => {
				return txt;
			});
			spanEl.blur();
		}
	}
</script>

<WidgetLabel {label}>
	<svelte:fragment slot="after">
		<span
			bind:textContent={value}
			bind:innerHTML={innterHTML}
			bind:this={spanEl}
			class="{label
				? 'mt-1.5'
				: ''} block overflow-auto resize-y py-2 px-3 w-full min-h-[42px] max-h-[500px] border border-gray-200 rounded-lg shadow-inner outline-none focus:ring-1 focus:ring-inset focus:ring-indigo-200 focus:shadow-inner dark:bg-gray-925"
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
