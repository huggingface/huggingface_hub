const SMOOTHING_FACTOR = 0.8;
const MINIMUM_VALUE = 0.00001;
registerProcessor(
    "vumeter",
    class extends AudioWorkletProcessor {
        _updateIntervalInMS;
        _sampleInFrames;
        _index;
        _buffer;

        constructor() {
            super();
            this._updateIntervalInMS = 50;
            this._sampleInFrames = parseInt((50.0 / 1000.0) * sampleRate);
            this._index = 0;
            this._buffer = new Float32Array(this._sampleInFrames);
        }

        process(inputs, outputs, parameters) {
            // Note that the input will be down-mixed to mono; however, if no inputs are
            // connected then zero channels will be passed in.
            // this._nextUpdateFrame -= inputs[0].length;
            // if (this._nextUpdateFrame < 0) {
            //     this._nextUpdateFrame += this.intervalInFrames;
            if (inputs.length > 0 && inputs[0].length > 0) {
                const rest = this._buffer.length - this._index;
                if (rest < inputs[0][0].length) {
                    this._buffer.set(inputs[0][0].slice(0, rest), this._index);
                    // console.log("buffer", this._buffer);
                    this.port.postMessage({
                        buffer: this._buffer.slice(0),
                        sampling_rate: sampleRate,
                    });
                    this._buffer.fill(0);
                    this._index = inputs[0][0].length - rest;
                } else {
                    this._buffer.set(inputs[0][0], this._index);
                    this._index += inputs[0][0].length;
                }
            }
            // }

            return true;
        }
    }
);
