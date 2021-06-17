<script>
	import { onDestroy, onMount } from "svelte";
	import IconMicrophone from "../../../Icons/IconMicrophone.svelte";
	import Recorder from "./Recorder";

	export let classNames = "";
	export let onRecordStart: () => void = () => null;
	export let onRecordStop: (blob: Blob) => void = () => null;

	let isRecording = false;
	let recorder: Recorder;

	onMount(() => {
		recorder = new Recorder();
	});

	onDestroy(() => {
		if (recorder) {
			recorder.stopRecording();
		}
	});

	async function onClick() {
		isRecording = !isRecording;
		if (isRecording) {
			recorder.start();
			onRecordStart();
		} else {
			const blob = await recorder.stopRecording();
			onRecordStop(blob);
		}
	}
</script>

<button class="btn-widget {classNames}" on:click={onClick} type="button">
	<div
		class="flex items-center {isRecording ? 'text-red-500 animate-pulse' : ''}"
	>
		<IconMicrophone classNames="-ml-1 mr-1.5" />
		<span>
			{isRecording ? "Click to stop recording" : "Record from browser"}
		</span>
	</div>
</button>
