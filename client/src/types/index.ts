// ── Transcription ──

export interface TranscriptEntry {
  id: string;
  timestamp: string;
  speaker: string;
  text: string;
  fact_checks: FactCheck[];
  /** Audio start offset (seconds) — used for diarization relabeling. */
  audio_start?: number;
  /** Audio end offset (seconds) — used for diarization relabeling. */
  audio_end?: number;
}

export interface FactCheck {
  claim: string;
  verdict: "supported" | "contradicted" | "uncertain";
  reasoning: string;
  fallacy?: string;
}

// ── Notes ──

export type NoteType = "action_item" | "decision" | "key_fact" | "risk";

export interface NoteEntry {
  id: string;
  type: NoteType;
  content: string;
  timestamp: string;
  owner?: string;
  due_date?: string;
  completed?: boolean;
}

// ── Participants ──

export interface Participant {
  speaker_label: string;
  display_name: string;
  role?: string;
  notes?: string;
}

// ── Entities ──

export interface PersonEntity {
  id: string;
  name: string;
  role?: string;
  mentioned_in: string[];
}

export interface ProjectEntity {
  id: string;
  name: string;
  description?: string;
  mentioned_in: string[];
}

export interface DecisionEntity {
  id: string;
  summary: string;
  date?: string;
  participants: string[];
  mentioned_in: string[];
}

export interface SessionEntities {
  people: PersonEntity[];
  projects: ProjectEntity[];
  decisions: DecisionEntity[];
}

// ── AI Events ──

export type AIEvent =
  | { type: "content_delta"; text: string }
  | { type: "tool_call"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: Record<string, unknown> }
  | { type: "done" }
  | { type: "error"; message: string };

// ── Session Settings (per-session overrides) ──

export interface SessionSettings {
  fact_checking?: boolean;
  suggestions?: boolean;
  notes?: boolean;
  search_transcript?: boolean;
  search_sessions?: boolean;
  web_search?: boolean;
  format_code?: boolean;
  deep_think?: "off" | "auto" | "always";
  diarization?: boolean;
  piiRedaction?: boolean;
  privacyMode?: boolean;
}

// ── Session ──

export type SessionStatus = "active" | "paused" | "ended";

export interface SessionSuggestion {
  id: string;
  text: string;
  timestamp: string;
}

export interface Session {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: SessionStatus;
  context?: string;
  transcript: TranscriptEntry[];
  notes: NoteEntry[];
  suggestions?: SessionSuggestion[];
  participants: Participant[];
  topics: string[];
  entities: SessionEntities;
  settings?: SessionSettings | null;
}

export interface SessionSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: SessionStatus;
  transcript_count: number;
  notes_count: number;
  topics: string[];
}

// ── Feature Toggles ──

export interface FeatureToggles {
  // AI features
  fact_checking: boolean;
  suggestions: boolean;
  notes: boolean;
  // AI tools
  search_transcript: boolean;
  search_sessions: boolean;
  web_search: boolean;
  format_code: boolean;
  // Deep think mode
  deep_think: "off" | "auto" | "always";
}

// ── Agent Activity Log ──

export type AgentLogType = "thinking" | "tool_call" | "tool_result" | "done" | "error";

export interface AgentLogEntry {
  id: string;
  timestamp: string;
  type: AgentLogType;
  name?: string;
  summary?: string;
  detail?: string;
}

// ── Audio Toggles ──

export interface AudioToggles {
  mic: boolean;
  system: boolean;
}

// ── Overlay Settings ──

export type OverlayMode = "topbar" | "cards";

export interface CardPosition {
  x: number;
  y: number;
}

export interface OverlaySettings {
  contentProtection: boolean;
  showTranscript: boolean;
  showFactChecks: boolean;
  showSuggestions: boolean;
  showNotes: boolean;
  overlayMode: OverlayMode;
  cardPositions: {
    transcript?: CardPosition;
    suggestion?: CardPosition;
    notes?: CardPosition;
  };
}

// ── Settings ──

export interface AppSettings {
  serverUrl: string;
  featureToggles: FeatureToggles;
  diarization: boolean;
  audioToggles: AudioToggles;
  overlaySettings: OverlaySettings;
  piiRedaction: boolean;
  privacyMode: boolean;
  // Client-local audio device selection (used when audio_capture_source === "client")
  micDeviceId: string | null;
  systemDeviceId: string | null;
}

// ── Presets ──

export interface Preset {
  id: string;
  name: string;
  description: string;
  default_tools: Record<string, boolean>;
}

// ── Server Config (from GET /api/config) ──

export interface LLMProviderConfig {
  configured: boolean;
  enabled: boolean;
  model: string;
  api_key_hint: string;
  api_base?: string;
}

export interface ServerConfig {
  server_platform: string;
  hostname: string;
  whisper_model: string;
  whisper_device: string;
  whisper_compute_type: string;
  whisper_language: string | null;
  provider_order: string[];
  audio_capture_source: "client" | "server";
  mic_device_id: string | null;
  system_device_id: string | null;
  ai_preset: string;
  custom_system_prompt: string | null;
  diarization_enabled: boolean;
  diarization_device: string | null;
  hf_diarization_token_hint: string;
  pii_redaction: boolean;
  privacy_mode: boolean;
  locked_settings: string[];
  llm_providers: Record<string, LLMProviderConfig>;
}

// ── Audio Devices ──

export interface AudioDeviceInfo {
  id: number;
  name: string;
  channels: number;
  sample_rate: number;
  is_input: boolean;
  is_output: boolean;
  is_loopback: boolean;
}

export interface ClientAudioDevice {
  deviceId: string;
  label: string;
}

// ── Electron API (exposed via preload) ──

// ── Insights Drawer ──

export type InsightsTab = "suggestions" | "notes" | "people" | "activity";

declare global {
  interface Window {
    electronAPI?: {
      toggleOverlay: () => Promise<boolean>;
      getAudioSources: () => Promise<{ id: string; name: string }[]>;
      setIgnoreMouseEvents: (ignore: boolean, forward: boolean) => void;
      setContentProtection: (enabled: boolean) => Promise<void>;
      onOverlayToggle: (callback: (isOverlay: boolean) => void) => () => void;
      setOverlayBounds: (bounds: { x?: number; y?: number; width?: number; height?: number }) => Promise<void>;
      getScreenSize: () => Promise<{ width: number; height: number }>;
      getPlatform: () => Promise<string>;
    };
  }
}
