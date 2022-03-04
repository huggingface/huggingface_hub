export default class Recorder {
	// see developers.google.com/web/updates/2016/01/mediarecorder
	type: "audio" | "video" = "audio";
	private apiToken: string | undefined;
	private audioContext: AudioContext;
	private isLoggedIn = false;
	private modelId: string;
	private onError: (err: string) => void;
	private renderText: (txt: string) => void;
	private renderWarning: (warning: string) => void;
	private socket: WebSocket;
	private stream: MediaStream;

	constructor(modelId: string, apiToken: string, renderText: (txt: string) => void, renderWarning: (warning: string) => void, onError: (err: string) => void){
		this.modelId = modelId;
		this.apiToken = apiToken;
		// TODO: for testing purposes, supply your hf.co/settings/tokens value in the line below
		this.apiToken = "";
		this.renderText = renderText;
		this.renderWarning = renderWarning;
		this.onError = onError;
	}

	async start() {
		if(!this.apiToken){
			throw new Error("You need to be loggedn in and have API token enabled. Find more at: hf.co/settings/token");
		}

		const constraints: MediaStreamConstraints =
			this.type === "video"
				? { audio: true, video: true }
				: { audio: true };
		this.stream = await navigator.mediaDevices.getUserMedia(constraints);

		this.socket = new WebSocket(`wss://api-inference.huggingface.co/asr/live/cpu/${this.modelId}`);

		this.socket.onerror = (_) => {
			this.onError("Webscoket connection error");
		}

		this.socket.onopen = (_) => {
			this.socket.send(`Bearer ${this.apiToken}`);
		}

		this.socket.onmessage = (e: MessageEvent) => {
			const data = JSON.parse(e.data);
			if(data.type === "status" && data.message === "Successful login"){
				this.isLoggedIn = true;
			}else{
				if(!!data.text){
					this.renderText(data.text)
				}else{
					this.renderWarning("result was empty");
				}
			}
		};

		this.audioContext = new AudioContext();
		await this.audioContext.audioWorklet.addModule("/audioProcessor.js");
		const microphone = this.audioContext.createMediaStreamSource(this.stream);
		const dataExtractor = new AudioWorkletNode(this.audioContext, "AudioDataExtractor");
		microphone.connect(dataExtractor).connect(this.audioContext.destination);

		dataExtractor.port.onmessage = (event) => {
			const {buffer, sampling_rate} = event.data;
			if(buffer.reduce((sum: number, x: number) => sum + x) === 0){
				this.renderWarning("🎤 input is empty: try speaking louder 🗣️ & make sure correct mic source is selected");
			}else{
				const base64: string = btoa(String.fromCharCode(...new Uint8Array(buffer.buffer)));
				const message = {
					raw: base64,
					sampling_rate,
				};
				if(this.isLoggedIn){
					try{
						this.socket.send(JSON.stringify(message));
					}catch(e){
						this.onError(`Error sending data to websocket: ${e}`);
					}
				}
			}
		};
	}

	stop() {
		this.isLoggedIn = false;
		this.audioContext?.close();
		this.socket?.close();
		this.stream?.getTracks().forEach((t) => t.stop());
	}
}
