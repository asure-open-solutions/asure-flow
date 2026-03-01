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

// AudioWorklet processor code (inline — registered as a blob URL)
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      // Convert float32 to int16
      const float32 = input[0];
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)));
      }
      this.port.postMessage({ pcmData: int16 }, [int16.buffer]);
    }
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
  onChunk: AudioChunkHandler | null = null;

  async start(options: AudioCaptureOptions = { mic: true, system: true }): Promise<void> {
    // Create AudioContext at target sample rate
    this.audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });

    // Register the worklet processor
    const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
    const workletUrl = URL.createObjectURL(blob);
    await this.audioContext.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);

    if (options.mic) {
      await this.enableMic(options.micDeviceId);
    }

    if (options.system) {
      await this.enableSystem().catch((err) => {
        console.warn("System audio capture unavailable:", err.message);
      });
    }
  }

  async enableMic(deviceId?: string): Promise<void> {
    if (this.micStream || !this.audioContext) return;

    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: TARGET_SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    });

    this.micSource = this.audioContext.createMediaStreamSource(this.micStream);
    this.micWorklet = new AudioWorkletNode(this.audioContext, "pcm-processor");
    this.micWorklet.port.onmessage = (e: MessageEvent) => {
      this.onChunk?.(STREAM_MIC, e.data.pcmData);
    };
    this.micSource.connect(this.micWorklet);
  }

  disableMic(): void {
    this.micWorklet?.disconnect();
    this.micSource?.disconnect();
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
    this.micSource = null;
    this.micWorklet = null;
  }

  async enableSystem(): Promise<void> {
    if (this.systemStream || !this.audioContext) return;

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
  }

  disableSystem(): void {
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
