export default class Recorder {
	// see developers.google.com/web/updates/2016/01/mediarecorder
	type: "audio" | "video" = "audio";
	private stream: MediaStream;
	private audioContext: AudioContext;
	private socket: WebSocket;
	private microphone: MediaStreamAudioSourceNode;
	private analyzer: AnalyserNode;
	private renderText: any;

	constructor(_renderText){
		this.audioContext = new AudioContext();
		this.audioContext.audioWorklet.addModule("/capture.js");
		this.analyzer = this.audioContext.createAnalyser();
		this.analyzer.fftSize = 2048;
		this.renderText = _renderText;
	}

	async start() {
		const constraints: MediaStreamConstraints =
			this.type === "video"
				? { audio: true, video: true }
				: { audio: true };
		this.stream = await navigator.mediaDevices.getUserMedia(constraints);
		this.socket = new WebSocket(
			"wss://api-inference.huggingface.co/wav"
		);
		console.log("start recording called")
	
		this.socket.onmessage = (e) => {
			// console.log(`Received ${e.data}`);
			const data = JSON.parse(e.data);
			this.renderText(data.text)
		};
	
		this.microphone = this.audioContext.createMediaStreamSource(this.stream);

		this.microphone.connect(this.analyzer);
	
		const node = new AudioWorkletNode(this.audioContext, "vumeter");
		node.port.onmessage = (event) => {
			const base64String = btoa(String.fromCharCode(...new Uint8Array(event.data.buffer.buffer)));
			const message = {
				raw: base64String,
				sampling_rate: event.data.sampling_rate,
			};
			this.socket.send(JSON.stringify(message));
		};
		this.microphone.connect(node).connect(this.audioContext.destination);
	}
	async stopRecording() {
		this.microphone?.disconnect();
		this.socket?.close();
		this.stream?.getTracks().forEach((t) => t.stop()); // Stop stream.
	}
	getAnalyzer(){
		return this.analyzer;
	}
}
