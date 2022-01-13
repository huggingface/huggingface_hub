<script>
	import { onDestroy, onMount, afterUpdate } from "svelte";
	import IconMagicWand from "../../../Icons/IconMagicWand.svelte";
	import Recorder from "./Recorder";

	export let classNames = "";
	// export let onRecordStart: () => void = () => null;
	export let onError: (err: string) => void = () => null;

	let canvasEl: HTMLCanvasElement;
	let analyzer: any;
	let bufferLength;
	let dataArray;

	let width = 0;
	let height = 0;

	let txt = ""

	let containerEl: HTMLElement;


	let isRecording = false;
	let recorder: Recorder;

	onMount(() => {
		recorder = new Recorder(updateTxt);
		analyzer = recorder.getAnalyzer();
		bufferLength = analyzer.frequencyBinCount;
  		dataArray = new Uint8Array(bufferLength);
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
				draw();
			} else {
				await recorder.stopRecording();
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

	function draw() {
		width = containerEl.clientWidth;
		height = containerEl.clientHeight;
		darwHelper();
	}

	function darwHelper() {
		if(!canvasEl){
			return;
		}
		const WIDTH = canvasEl.width
		const HEIGHT = canvasEl.height;

		requestAnimationFrame(darwHelper);

		analyzer.getByteTimeDomainData(dataArray);

		const canvasCtx = canvasEl.getContext("2d");

		canvasCtx.fillStyle = 'rgb(200, 200, 200)';
		canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

		canvasCtx.lineWidth = 2;
		canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

		canvasCtx.beginPath();

		let sliceWidth = WIDTH * 1.0 / bufferLength;
		let x = 0;


		for(let i = 0; i < bufferLength; i++) {

		let v = dataArray[i] / 128.0;
		let y = v * HEIGHT/2;

		if(i === 0) {
			canvasCtx.moveTo(x, y);
		} else {
			canvasCtx.lineTo(x, y);
		}

		x += sliceWidth;
		}

		canvasCtx.lineTo(canvasEl.width, canvasEl.height/2);
		canvasCtx.stroke();

	}

	function updateTxt(_txt){
		txt = _txt;
	}

	afterUpdate(draw);
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
