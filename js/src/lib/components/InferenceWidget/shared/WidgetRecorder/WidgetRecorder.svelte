<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import IconMicrophone from "../../../Icons/IconMicrophone.svelte";
	import Recorder from "./Recorder";

	export let classNames = "";
	export let onRecordStart: () => void = () => null;
	export let onRecordStop: (blob: Blob) => void = () => null;
	export let onError: (err: string) => void = () => null;

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
		try {
			isRecording = !isRecording;
			if (isRecording) {
				await recorder.start();
				onRecordStart();
			} else {
				const blob = await recorder.stopRecording();
				onRecordStop(blob);
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
					onError(`Encountered error "${e.name}: ${e.message}"`);
					break;
				}
			}
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
