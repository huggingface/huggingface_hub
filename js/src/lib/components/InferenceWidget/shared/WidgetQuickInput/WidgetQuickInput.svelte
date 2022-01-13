<script>
	import WidgetSubmitBtn from "../WidgetSubmitBtn/WidgetSubmitBtn.svelte";

	export let flatTop = false;
	export let isLoading: boolean;
	export let onClickSubmitBtn: (e: MouseEvent) => void;
	export let placeholder = "Your sentence here...";
	export let submitButtonLabel: string | undefined = undefined;
	export let value: string = "";
</script>

<div class="flex h-10">
	<input
		bind:value
		class="form-input-alt flex-1 rounded-r-none min-w-0 {flatTop
			? 'rounded-t-none'
			: ''}"
		{placeholder}
		required={true}
		type="text"
	/>
	<WidgetSubmitBtn
		classNames="rounded-l-none border-l-0 {flatTop ? 'rounded-t-none' : ''}"
		{isLoading}
		label={submitButtonLabel}
		onClick={onClickSubmitBtn}
	/>
</div>
