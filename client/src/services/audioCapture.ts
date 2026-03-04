/**
 * Dual audio capture — mic + system audio via AudioWorklet.
 *
 * Captures both streams as 16 kHz mono PCM and sends chunks to a callback.
 * Supports independent enable/disable of mic and system streams mid-session.
 */

const TARGET_SAMPLE_RATE = 16000;
const STREAM_MIC = 0;
const STREAM_SYSTEM = 1;

export type AudioChunkHandler = (streamId: number, pcmData: Int16Array) => void;

export interface AudioCaptureOptions {
  mic: boolean;
  system: boolean;
  micDeviceId?: string;
}

/** Result from start() indicating what actually started. */
export interface AudioCaptureResult {
  mic: boolean;
  system: boolean;
  micError?: string;
  systemError?: string;
}

// AudioWorklet processor code (inline — registered as a blob URL)
// Resamples from the AudioContext's native sample rate down to 16 kHz
// before converting float32 → int16 PCM and posting to the main thread.
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._resampleRatio = sampleRate / 16000;
    this._resampleBuffer = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const float32 = input[0];

    // Fast path: no resampling needed
    if (sampleRate === 16000) {
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)));
      }
      this.port.postMessage({ pcmData: int16 }, [int16.buffer]);
      return true;
    }

    // Accumulate input and downsample to 16 kHz
    const prev = this._resampleBuffer;
    const combined = new Float32Array(prev.length + float32.length);
    combined.set(prev);
    combined.set(float32, prev.length);

    const ratio = this._resampleRatio;
    const outLen = Math.floor(combined.length / ratio);
    if (outLen === 0) {
      this._resampleBuffer = combined;
      return true;
    }

    const int16 = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const idx = Math.floor(srcIdx);
      const frac = srcIdx - idx;
      // Linear interpolation
      const s0 = combined[idx];
      const s1 = idx + 1 < combined.length ? combined[idx + 1] : s0;
      const sample = s0 + frac * (s1 - s0);
      int16[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
    }

    // Keep unconsumed tail for next call
    const consumed = Math.floor(outLen * ratio);
    this._resampleBuffer = combined.slice(consumed);

    this.port.postMessage({ pcmData: int16 }, [int16.buffer]);
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
`;

export class AudioCapture {
  private audioContext: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private systemStream: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private systemSource: MediaStreamAudioSourceNode | null = null;
  private micWorklet: AudioWorkletNode | null = null;
  private systemWorklet: AudioWorkletNode | null = null;
  private micEnabling = false;
  private systemEnabling = false;
  private lastMicDeviceId: string | undefined;
  private pendingMicSwitch: { deviceId?: string } | null = null;
  onChunk: AudioChunkHandler | null = null;

  async start(options: AudioCaptureOptions = { mic: true, system: true }): Promise<AudioCaptureResult> {
    const result: AudioCaptureResult = { mic: false, system: false };

    // Create AudioContext at the device's native sample rate.
    // Resampling to 16 kHz is handled inside the AudioWorklet to avoid
    // silence on platforms where forcing a non-native rate causes
    // MediaStreamAudioSourceNode to output zeros (e.g. macOS).
    this.audioContext = new AudioContext();

    // Register the worklet processor
    const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
    const workletUrl = URL.createObjectURL(blob);
    try {
      await this.audioContext.audioWorklet.addModule(workletUrl);
    } finally {
      URL.revokeObjectURL(workletUrl);
    }

    if (options.mic) {
      try {
        await this.enableMic(options.micDeviceId);
        result.mic = this.hasMic;
      } catch (err) {
        result.micError = err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone permission denied"
          : `Microphone error: ${(err as Error).message}`;
        console.warn(result.micError);
      }
    }

    if (options.system) {
      try {
        await this.enableSystem();
        result.system = this.hasSystem;
      } catch (err) {
        result.systemError = err instanceof DOMException && err.name === "NotAllowedError"
          ? "System audio permission denied"
          : `System audio error: ${(err as Error).message}`;
        console.warn(result.systemError);
      }
    }

    // If nothing is capturing, tear down the AudioContext
    if (!result.mic && !result.system) {
      this.audioContext.close();
      this.audioContext = null;
    }

    return result;
  }

  async enableMic(deviceId?: string): Promise<void> {
    if (!this.audioContext) return;

    // If already connected with a different device, tear down first
    if (this.micStream && deviceId !== undefined && deviceId !== this.lastMicDeviceId) {
      this.disableMic();
    }

    // If another enableMic is in-flight, queue this request
    if (this.micEnabling) {
      this.pendingMicSwitch = { deviceId };
      return;
    }

    // Already connected with the same device
    if (this.micStream) return;

    this.micEnabling = true;
    this.pendingMicSwitch = null;

    if (deviceId !== undefined) this.lastMicDeviceId = deviceId;
    const useDeviceId = deviceId ?? this.lastMicDeviceId;

    try {
      // Ensure AudioContext is running — Chrome may suspend it when all sources are removed
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      this.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          ...(useDeviceId ? { deviceId: { exact: useDeviceId } } : {}),
        },
      });

      this.micSource = this.audioContext.createMediaStreamSource(this.micStream);
      this.micWorklet = new AudioWorkletNode(this.audioContext, "pcm-processor");
      this.micWorklet.port.onmessage = (e: MessageEvent) => {
        this.onChunk?.(STREAM_MIC, e.data.pcmData);
      };
      this.micSource.connect(this.micWorklet);
    } finally {
      this.micEnabling = false;
    }

    // Process queued device switch
    if (this.pendingMicSwitch !== null) {
      const pending = this.pendingMicSwitch;
      this.pendingMicSwitch = null;
      this.disableMic();
      await this.enableMic(pending.deviceId);
    }
  }

  disableMic(): void {
    this.micEnabling = false;
    this.micWorklet?.disconnect();
    this.micSource?.disconnect();
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
    this.micSource = null;
    this.micWorklet = null;
  }

  async enableSystem(): Promise<void> {
    if (this.systemStream || this.systemEnabling || !this.audioContext) return;
    this.systemEnabling = true;

    try {
      // Ensure AudioContext is running — Chrome may suspend it when all sources are removed
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      // Use getDisplayMedia to capture system audio
      this.systemStream = await navigator.mediaDevices.getDisplayMedia({
        video: true, // Required by API
        audio: true,
      });

      // Remove video tracks — we only want audio
      for (const track of this.systemStream.getVideoTracks()) {
        track.stop();
        this.systemStream.removeTrack(track);
      }

      // Check if we actually got audio tracks
      if (this.systemStream.getAudioTracks().length === 0) {
        console.warn("No audio track in system capture");
        this.systemStream = null;
        return;
      }

      this.systemSource = this.audioContext.createMediaStreamSource(this.systemStream);
      this.systemWorklet = new AudioWorkletNode(this.audioContext, "pcm-processor");
      this.systemWorklet.port.onmessage = (e: MessageEvent) => {
        this.onChunk?.(STREAM_SYSTEM, e.data.pcmData);
      };
      this.systemSource.connect(this.systemWorklet);
    } finally {
      this.systemEnabling = false;
    }
  }

  disableSystem(): void {
    this.systemEnabling = false;
    this.systemWorklet?.disconnect();
    this.systemSource?.disconnect();
    this.systemStream?.getTracks().forEach((t) => t.stop());
    this.systemStream = null;
    this.systemSource = null;
    this.systemWorklet = null;
  }

  stop(): void {
    this.disableMic();
    this.disableSystem();
    this.audioContext?.close();
    this.audioContext = null;
  }

  get hasMic(): boolean {
    return (this.micStream?.getAudioTracks().length ?? 0) > 0;
  }

  get hasSystem(): boolean {
    return (this.systemStream?.getAudioTracks().length ?? 0) > 0;
  }
}
