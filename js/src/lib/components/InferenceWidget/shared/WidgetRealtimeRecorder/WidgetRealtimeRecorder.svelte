<script>
	import type { WidgetProps } from "../../shared/types";
	import { onDestroy, onMount } from "svelte";
	import IconMagicWand from "../../../Icons/IconMagicWand.svelte";
	import Recorder from "./Recorder";
	export let apiToken: WidgetProps["apiUrl"];
	export let classNames = "";
	export let model: WidgetProps["model"];
	export let onError: (err: string) => void = () => null;
	let isRecording = false;
	let recorder: Recorder;
	let txt = "";
	let warning = "";
	async function onClick() {
		try {
			isRecording = !isRecording;
			if (isRecording) {
				await recorder.start();
			} else {
				await recorder.stop();
			}
		} catch (e) {
			isRecording = false;
			switch (e.name) {
				case "NotAllowedError": {
					onError("Please allow access to your microphone");
					break;
				}
				case "NotFoundError": {
					onError("No microphone found on your device");
					break;
				}
				default: {
					onError(`${e.name}: ${e.message}`);
					break;
				}
			}
		}
	}
	function renderText(_txt) {
		warning = "";
		txt = _txt;
		onError("");
	}
	function renderWarning(_warning) {
		warning = _warning;
	}
	onMount(() => {
		recorder = new Recorder(
			model.id,
			apiToken,
			renderText,
			renderWarning,
			onError
		);
	});
	onDestroy(() => {
		if (recorder) {
			recorder.stop();
		}
	});
</script>

<button class="btn-widget {classNames}" on:click={onClick} type="button">
	<div
		class="flex items-center {isRecording ? 'text-red-500 animate-pulse' : ''}"
	>
		<IconMagicWand classNames="-ml-1 mr-1.5" />
		<span>
			{isRecording ? "Stop speech recognition" : "Realtime speech recognition"}
		</span>
	</div>
</button>

{#if isRecording}
	<div
		class="relative top-0 left-0 inline-flex w-full mb-2 mt-4 items-center justify-center {!!warning &&
			'animate-pulse'}"
	>
		{#if warning}
			<p class="opacity-50">{warning}</p>
		{:else}
			<p class="lowercase font-mono">{txt}</p>
		{/if}
	</div>
{/if}
