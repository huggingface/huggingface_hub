<script lang="ts">
	import IconSpin from "../../../Icons/IconSpin.svelte";
	import IconFile from "../../../Icons/IconFile.svelte";

	export let accept: string | undefined;
	export let classNames = "";
	export let isLoading = false;
	export let label = "Browse for file";
	export let onSelectFile: (file: File | Blob) => void;

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
	<label
		class="btn-widget {isDragging ? 'ring' : ''} {isLoading
			? 'text-gray-600'
			: ''}"
	>
		{#if isLoading}
			<IconSpin classNames="-ml-1 mr-1.5 text-gray-600 animate-spin" />
		{:else}
			<IconFile classNames="-ml-1 mr-1.5" />
		{/if}
		<input
			{accept}
			bind:this={fileInput}
			on:change={onChange}
			disabled={isLoading}
			style="display: none;"
			type="file"
		/>
		{label}
	</label>
</div>
