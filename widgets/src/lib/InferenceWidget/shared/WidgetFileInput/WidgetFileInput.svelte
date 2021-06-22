<script>
	import IconFile from "../../../Icons/IconFile.svelte";

	export let accept: string | undefined;
	export let classNames = "";
	export let label = "Browse for file";
	export let onSelectFile: (file: File) => void;

	let fileInput: HTMLInputElement;
	let isDragging = false;

	function onChange() {
		const file = fileInput.files?.[0];
		if (file) {
			onSelectFile(file);
		}
	}
</script>

<div
	class={classNames}
	on:dragenter={() => {
		isDragging = true;
	}}
	on:dragover|preventDefault
	on:dragleave={() => {
		isDragging = false;
	}}
	on:drop|preventDefault={(e) => {
		isDragging = false;
		fileInput.files = e.dataTransfer?.files ?? null;
		onChange();
	}}
>
	<label class="btn-widget {isDragging ? 'ring' : ''}">
		<IconFile classNames="-ml-1 mr-1.5" />
		<input
			{accept}
			bind:this={fileInput}
			on:change={onChange}
			style="display: none;"
			type="file"
		/>
		{label}
	</label>
</div>
