import { delay } from "../../../../utils/ViewUtils";

export default class Recorder {
	// see developers.google.com/web/updates/2016/01/mediarecorder
	type: "audio" | "video" = "audio";
	private stream: MediaStream;
	private mediaRecorder: MediaRecorder;
	private recordedBlobs: Blob[] = [];
	public outputBlob?: Blob;

	get desiredMimeType(): string {
		return this.type === "video" ? "video/webm" : "audio/webm";
	}
	get mimeType() {
		return this.mediaRecorder.mimeType;
	}
	async start() {
		this.recordedBlobs = [];

		const constraints: MediaStreamConstraints =
			this.type === "video"
				? { audio: true, video: true }
				: { audio: true };
		this.stream = await navigator.mediaDevices.getUserMedia(constraints);
		this.startRecording();
	}
	private startRecording() {
		this.outputBlob = undefined;
		this.mediaRecorder = new MediaRecorder(this.stream, {
			mimeType: this.desiredMimeType,
		});
		this.mediaRecorder.onstop = this.handleStop.bind(this);
		this.mediaRecorder.ondataavailable =
			this.handleDataAvailable.bind(this);
		this.mediaRecorder.start(10); // timeslice in ms
	}
	handleStop() {}
	handleDataAvailable(evt: any) {
		if (evt.data && evt.data.size > 0) {
			this.recordedBlobs.push(evt.data);
		}
	}
	async stopRecording(): Promise<Blob> {
		if (this.mediaRecorder) {
			this.mediaRecorder.stop();
		}
		if (this.stream) {
			this.stream.getTracks().forEach((t) => t.stop()); // Stop stream.
		}

		await delay(30);
		// Wait for the last blob in handleDataAvailable.
		// Alternative: hook into `onstop` event.
		const superBuffer = new Blob(this.recordedBlobs, {
			type: this.mimeType,
		});
		this.outputBlob = superBuffer;
		return superBuffer;
	}
}
