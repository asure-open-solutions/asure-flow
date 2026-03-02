/**
 * WebSocket client — audio streaming and session events.
 */

import type { AIEvent, FeatureToggles, Participant } from "@/types";
import { getServerUrl } from "./api";

type TranscriptionHandler = (entry: {
  speaker: string;
  text: string;
  start: number;
  end: number;
}) => void;

type RelabelHandler = (data: { entry_id: string; speaker: string }) => void;

type AIEventHandler = (event: AIEvent) => void;

type SpeakerRenamedHandler = (data: {
  speaker_label: string;
  display_name: string;
  participant: Participant;
}) => void;

// ── Audio WebSocket ──

export class AudioWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private disposed = false;
  onTranscription: TranscriptionHandler | null = null;
  onRelabel: RelabelHandler | null = null;
  onConnectionChange: ((connected: boolean) => void) | null = null;

  connect() {
    const wsUrl = getServerUrl().replace(/^http/, "ws") + "/ws/audio";
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.onConnectionChange?.(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "transcription" && this.onTranscription) {
          this.onTranscription(data);
        } else if (data.type === "relabel" && this.onRelabel) {
          this.onRelabel(data);
        }
      } catch {
        // Ignore non-JSON messages
      }
    };

    this.ws.onclose = () => {
      this.onConnectionChange?.(false);
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  /** Send a PCM audio chunk with a stream ID prefix byte. */
  sendAudio(streamId: number, pcmData: Int16Array) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const header = new Uint8Array([streamId]);
    const payload = new Uint8Array(header.length + pcmData.byteLength);
    payload.set(header, 0);
    payload.set(new Uint8Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength), 1);
    this.ws.send(payload.buffer);
  }

  disconnect() {
    this.disposed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  private scheduleReconnect() {
    if (this.disposed) return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}

// ── Session WebSocket ──

export class SessionWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private disposed = false;
  private sessionId: string;
  onAIEvent: AIEventHandler | null = null;
  onSpeakerRenamed: SpeakerRenamedHandler | null = null;
  onConnectionChange: ((connected: boolean) => void) | null = null;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
  }

  connect() {
    const wsUrl = getServerUrl().replace(/^http/, "ws") + `/ws/session/${this.sessionId}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.onConnectionChange?.(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "ai_event" && this.onAIEvent) {
          this.onAIEvent(data.event);
        } else if (data.type === "speaker_renamed" && this.onSpeakerRenamed) {
          this.onSpeakerRenamed(data);
        }
      } catch {
        // Ignore
      }
    };

    this.ws.onclose = () => {
      this.onConnectionChange?.(false);
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  /** Forward a transcription entry to the session for AI processing. */
  sendTranscription(entryId: string, speaker: string, text: string, start?: number, end?: number) {
    this.send({ type: "transcription", entry_id: entryId, speaker, text, audio_start: start, audio_end: end });
  }

  /** Forward a speaker relabel to persist in the server session model. */
  sendRelabel(entryId: string, speaker: string) {
    this.send({ type: "relabel", entry_id: entryId, speaker });
  }

  /** Rename a speaker (e.g., "Speaker 1" → "Alice"). */
  sendRenameSpeaker(speakerLabel: string, displayName: string, role?: string) {
    this.send({ type: "rename_speaker", speaker_label: speakerLabel, display_name: displayName, role });
  }

  /** Update feature toggles. */
  sendConfig(toggles: FeatureToggles) {
    this.send({ type: "config", ...toggles });
  }

  /** Update session context (user briefing). */
  sendSessionContext(context: string) {
    this.send({ type: "config", session_context: context });
  }

  /** Re-trigger agent on recent context (after edit, delete, toggle change, reconnect). */
  sendRerun() {
    this.send({ type: "rerun" });
  }

  /** End the session. */
  endSession() {
    this.send({ type: "end_session" });
  }

  disconnect() {
    this.disposed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  private send(data: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private scheduleReconnect() {
    if (this.disposed) return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}
