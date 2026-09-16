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

type SessionServerEvent =
  | { type: "error" | "warning"; message: string }
  | { type: "session_saved" | "session_ended" | "heartbeat" };

// ── Audio WebSocket ──

export class AudioWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private pendingAudio: ArrayBuffer[] = [];
  private readonly maxPendingChunks = 250;
  private disposed = false;
  private captureLocation: "auto" | "client" | "server";
  onTranscription: TranscriptionHandler | null = null;
  onRelabel: RelabelHandler | null = null;
  onConnectionChange: ((connected: boolean) => void) | null = null;

  constructor(captureLocation: "auto" | "client" | "server" = "auto") {
    this.captureLocation = captureLocation;
  }

  connect() {
    if (
      this.ws?.readyState === WebSocket.OPEN ||
      this.ws?.readyState === WebSocket.CONNECTING
    ) return;
    this.disposed = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const query = this.captureLocation === "auto"
      ? ""
      : `?capture=${encodeURIComponent(this.captureLocation)}`;
    const wsUrl = getServerUrl().replace(/^http/, "ws") + `/ws/audio${query}`;
    const ws = new WebSocket(wsUrl);
    this.ws = ws;
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.reconnectDelay = 1000;
      this.onConnectionChange?.(true);
      for (const payload of this.pendingAudio.splice(0)) {
        ws.send(payload);
      }
    };

    ws.onmessage = (event) => {
      if (this.ws !== ws) return;
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

    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.onConnectionChange?.(false);
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      if (this.ws === ws) ws.close();
    };
  }

  /** Send a PCM audio chunk with a stream ID prefix byte. */
  sendAudio(streamId: number, pcmData: Int16Array) {
    const header = new Uint8Array([streamId]);
    const payload = new Uint8Array(header.length + pcmData.byteLength);
    payload.set(header, 0);
    payload.set(new Uint8Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength), 1);
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(payload.buffer);
      return;
    }
    this.pendingAudio.push(payload.buffer);
    if (this.pendingAudio.length > this.maxPendingChunks) {
      this.pendingAudio.splice(0, this.pendingAudio.length - this.maxPendingChunks);
    }
  }

  disconnect() {
    this.disposed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
    this.pendingAudio = [];
  }

  private scheduleReconnect() {
    if (this.disposed) return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    const delay = this.reconnectDelay * (0.8 + Math.random() * 0.4);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}

// ── Session WebSocket ──

export class SessionWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private pendingMessages: string[] = [];
  private readonly maxPendingMessages = 1000;
  private disposed = false;
  private sessionId: string;
  onAIEvent: AIEventHandler | null = null;
  onSpeakerRenamed: SpeakerRenamedHandler | null = null;
  onServerEvent: ((event: SessionServerEvent) => void) | null = null;
  onConnectionChange: ((connected: boolean) => void) | null = null;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
  }

  connect() {
    if (
      this.ws?.readyState === WebSocket.OPEN ||
      this.ws?.readyState === WebSocket.CONNECTING
    ) return;
    this.disposed = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const wsUrl = getServerUrl().replace(/^http/, "ws") + `/ws/session/${this.sessionId}`;
    const ws = new WebSocket(wsUrl);
    this.ws = ws;

    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.reconnectDelay = 1000;
      this.onConnectionChange?.(true);
      for (const message of this.pendingMessages.splice(0)) {
        ws.send(message);
      }
    };

    ws.onmessage = (event) => {
      if (this.ws !== ws) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === "ai_event" && this.onAIEvent) {
          this.onAIEvent(data.event);
        } else if (data.type === "speaker_renamed" && this.onSpeakerRenamed) {
          this.onSpeakerRenamed(data);
        } else if (
          data.type === "error" ||
          data.type === "warning" ||
          data.type === "session_saved" ||
          data.type === "session_ended" ||
          data.type === "heartbeat"
        ) {
          this.onServerEvent?.(data);
        }
      } catch {
        // Ignore
      }
    };

    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.onConnectionChange?.(false);
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      if (this.ws === ws) ws.close();
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

  /** Re-trigger agent on recent context after an explicit state change. */
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
    this.pendingMessages = [];
  }

  private send(data: Record<string, unknown>) {
    const message = JSON.stringify(data);
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(message);
      return;
    }
    this.pendingMessages.push(message);
    if (this.pendingMessages.length > this.maxPendingMessages) {
      this.pendingMessages.splice(0, this.pendingMessages.length - this.maxPendingMessages);
    }
  }

  private scheduleReconnect() {
    if (this.disposed) return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    const delay = this.reconnectDelay * (0.8 + Math.random() * 0.4);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}
