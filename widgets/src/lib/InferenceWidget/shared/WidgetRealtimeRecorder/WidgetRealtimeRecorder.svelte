<script>
	import { onDestroy, onMount, afterUpdate } from "svelte";
	import IconMagicWand from "../../../Icons/IconMagicWand.svelte";
	import Recorder from "./Recorder";

	export let classNames = "";
	// export let onRecordStart: () => void = () => null;
	export let onError: (err: string) => void = () => null;

	// vars for handling Recorder
	let txt = "";
	let isRecording = false;
	let recorder: Recorder;

	// vars for visualizing audio
	let containerEl: HTMLElement;
	let canvasEl: HTMLCanvasElement;
	let width = 0;
	let height = 0;
	let analyzer: AnalyserNode;
	let bufferLength: number;
	let dataArray: Uint8Array;

	async function onClick() {
		try {
			isRecording = !isRecording;
			if (isRecording) {
				await recorder.start();
				drawCanvas();
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
					onError(`Encountered error "${e.name}: ${e.message}"`);
					break;
				}
			}
		}
	}

	function drawCanvas() {
		width = containerEl.clientWidth;
		height = containerEl.clientHeight;
		darwCanvasHelper();
	}

	function darwCanvasHelper() {
		if (!canvasEl) {
			return;
		}
		const WIDTH = canvasEl.width;
		const HEIGHT = canvasEl.height;

		requestAnimationFrame(darwCanvasHelper);

		analyzer.getByteTimeDomainData(dataArray);

		const canvasCtx = canvasEl.getContext("2d");

		canvasCtx.fillStyle = "rgb(200, 200, 200)";
		canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
		canvasCtx.lineWidth = 2;
		canvasCtx.strokeStyle = "rgb(0, 0, 0)";
		canvasCtx.beginPath();

		let sliceWidth = (WIDTH * 1.0) / bufferLength;
		let x = 0;

		for (let i = 0; i < bufferLength; i++) {
			let v = dataArray[i] / 128.0;
			let y = (v * HEIGHT) / 2;

			if (i === 0) {
				canvasCtx.moveTo(x, y);
			} else {
				canvasCtx.lineTo(x, y);
			}

			x += sliceWidth;
		}

		canvasCtx.lineTo(canvasEl.width, canvasEl.height / 2);
		canvasCtx.stroke();
	}

	function renderTextCallback(_txt) {
		txt = _txt;
	}

	// svelte lifecycle functions

	onMount(() => {
		recorder = new Recorder(renderTextCallback);
		analyzer = recorder.getAnalyzer();
		bufferLength = analyzer.frequencyBinCount;
		dataArray = new Uint8Array(bufferLength);
	});

	afterUpdate(drawCanvas);

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

<div
	class="relative top-0 left-0 inline-flex w-full h-28 mb-2 mt-4"
	bind:this={containerEl}
>
	<canvas class="absolute top-0 left-0" bind:this={canvasEl} {width} {height} />
	<div class="relative top-0 left-0 flex items-center justify-center w-full">
		<p>{txt}</p>
	</div>
</div>
