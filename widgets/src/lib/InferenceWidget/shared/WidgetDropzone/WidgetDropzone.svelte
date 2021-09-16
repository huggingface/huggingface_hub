<script>
	import IconSpin from "../../../Icons/IconSpin.svelte";
	import { proxify } from "../../shared/helpers";

	export let accept = "image/*";
	export let classNames = "";
	export let isLoading = false;
	export let label = "Drag image file here or click to browse from your device";
	export let onSelectFile: (file: File | Blob) => void;
	export let onError: (e: string) => void;

	let fileInput: HTMLInputElement;
	let isDragging = false;
	let imgSrc = "";

	function onChange() {
		const file = fileInput.files?.[0];
		if (file) {
			imgSrc = URL.createObjectURL(file);
			onSelectFile(file);
		}
	}

	async function onDrop(e: DragEvent) {
		isDragging = false;
		const itemList = e.dataTransfer?.items;
		if (!itemList) {
			return;
		}
		const items: DataTransferItem[] = [];
		for (let i = 0; i < itemList.length; i++) {
			items.push(itemList[i]);
		}
		const uriItem = items.find(
			(x) => x.kind === "string" && x.type === "text/uri-list"
		);
		const fileItem = items.find((x) => x.kind === "file");
		if (uriItem) {
			const url = await new Promise<string>((resolve) =>
				uriItem.getAsString((s) => resolve(s))
			);
			const proxiedUrl = proxify(url);
			const res = await fetch(proxiedUrl);
			const file = await res.blob();

			imgSrc = URL.createObjectURL(file);
			onSelectFile(file);
		} else if (fileItem) {
			const file = fileItem.getAsFile();
			if (file) {
				imgSrc = URL.createObjectURL(file);
				onSelectFile(file);
			}
		} else {
			onError(`Unrecognized dragged and dropped file or element.`);
		}
	}
</script>

<input
	{accept}
	bind:this={fileInput}
	on:change={onChange}
	style="display: none;"
	type="file"
/>
<div
	class="relative border-2 border-dashed rounded mb-2 px-3 py-7 text-center cursor-pointer {isDragging
		? 'border-green-300 bg-green-50 text-green-500'
		: 'text-gray-500'} {classNames}"
	on:click={() => {
		fileInput.click();
	}}
	on:dragenter={() => {
		isDragging = true;
	}}
	on:dragleave={() => {
		isDragging = false;
	}}
	on:dragover|preventDefault
	on:drop|preventDefault={onDrop}
>
	{#if !imgSrc}
		<span class="pointer-events-none text-sm">{label}</span>
	{:else}
		<slot />
	{/if}
	{#if isLoading}
		<div
			class="absolute flex items-center justify-center top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-12 w-12 bg-white border border-gray-100 rounded-full shadow"
		>
			<IconSpin classNames="text-purple-500 animate-spin h-6 w-6" />
		</div>
	{/if}
</div>
