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

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
}

export type AIEvent =
  | { type: "content_delta"; text: string; specialist?: string }
  | { type: "tool_call"; name: string; arguments: Record<string, unknown>; specialist?: string }
  | { type: "tool_result"; name: string; result: Record<string, unknown>; specialist?: string }
  | { type: "done"; reason?: string; usage?: TokenUsage }
  | { type: "preempted" }
  | { type: "error"; message: string; specialist?: string };

// ── Session Settings (per-session overrides) ──

export interface SessionSettings {
  fact_checking?: boolean;
  suggestions?: boolean;
  notes?: boolean;
  search_transcript?: boolean;
  search_sessions?: boolean;
  web_search?: boolean;
  format_code?: boolean;
  deep_think?: "off" | "auto" | "always" | null;
  agent_mode?: "unified" | "specialists" | null;
  parallel_tools?: boolean;
  diarization?: boolean;
  pii_redaction?: boolean;
  privacy_mode?: boolean;
}

// ── Session ──

export type SessionStatus = "active" | "paused" | "ended";

export interface SessionSuggestion {
  id: string;
  text: string;
  timestamp: string;
  responding_to?: string;
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
  token_usage?: TokenUsage;
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
  // Agent execution mode
  agent_mode: "unified" | "specialists";
  parallel_tools: boolean;
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
  specialist?: string;
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

// ── User Profile (from GET /api/profile) ──
// Tier 2: portable user preferences, synced across all clients via the server.

export interface UserProfile {
  // Feature toggles
  fact_checking: boolean;
  suggestions: boolean;
  notes: boolean;
  search_transcript: boolean;
  search_sessions: boolean;
  web_search: boolean;
  format_code: boolean;
  deep_think: "off" | "auto" | "always";
  // Agent execution mode
  agent_mode: "unified" | "specialists";
  parallel_tools: boolean;
  // AI behaviour
  ai_preset: string;
  custom_system_prompt: string | null;
  // Diarization preference
  diarization_enabled: boolean;
  // Safety
  pii_redaction: boolean;
  privacy_mode: boolean;
}

// ── Settings ──

export interface AppSettings {
  // Tier 3: device-local (localStorage only — never synced to server)
  serverUrl: string;
  audioToggles: AudioToggles;
  overlaySettings: OverlaySettings;
  micDeviceId: string | null;
  systemDeviceId: string | null;

  // Tier 2: user profile (server-synced, cached in localStorage for fast startup)
  featureToggles: FeatureToggles;
  diarization: boolean;
  piiRedaction: boolean;
  privacyMode: boolean;
  aiPreset: string;
  customSystemPrompt: string | null;
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
  id: string;
  name: string;
  litellm_prefix: string;
  model: string;
  api_key_hint: string;
  api_base: string;
  configured: boolean;
  enabled: boolean;
}

export interface ServerConfig {
  // Server identity
  server_platform: string;
  hostname: string;
  // Transcription hardware
  whisper_model: string;
  whisper_device: string;
  whisper_compute_type: string;
  whisper_language: string | null;
  // LLM routing
  routing_strategy: string;
  // Audio capture mode + server-side device IDs (only relevant when audio_capture_source="server")
  audio_capture_source: "client" | "server";
  /** Server microphone device (used when audio_capture_source="server"). */
  mic_device_id: string | null;
  /** Set when server captures system audio via loopback — client should not duplicate. */
  system_device_id: string | null;
  // Diarization hardware
  diarization_device: string | null;
  hf_diarization_token_hint: string;
  // VAD / speed
  vad_silence_ms: number;
  vad_min_buffer_sec: number;
  // Admin
  locked_settings: string[];
  // Providers (ordered array — position = fallback priority)
  llm_providers: LLMProviderConfig[];
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
      getEnvServerUrl: () => string | null;
      setIgnoreMouseEvents: (ignore: boolean, forward: boolean) => void;
      setContentProtection: (enabled: boolean) => Promise<void>;
      setOverlayBounds: (bounds: { x?: number; y?: number; width?: number; height?: number }) => Promise<void>;
      getScreenSize: () => Promise<{ width: number; height: number }>;
      getPlatform: () => Promise<string>;

      // Window controls
      windowMinimize: () => void;
      windowMaximize: () => void;
      windowClose: () => void;
      windowIsMaximized: () => Promise<boolean>;
      onMaximizeChange: (callback: (maximized: boolean) => void) => () => void;

      // Overlay state sync
      sendOverlaySync: (data: unknown) => void;
      onOverlaySync: (callback: (data: unknown) => void) => () => void;
      onOverlayOpened: (callback: () => void) => () => void;
      onOverlayToggle: (callback: (isOverlay: boolean) => void) => () => void;

      // Recording toggle (overlay ↔ main window)
      toggleRecording: () => void;
      onToggleRecording: (callback: () => void) => () => void;

      // Audio toggle (overlay → main window)
      setAudioToggle: (toggle: { mic?: boolean; system?: boolean }) => void;
      onAudioToggle: (callback: (toggle: { mic?: boolean; system?: boolean }) => void) => () => void;

      // Suggestion focus (overlay ↔ main window)
      focusSuggestion: (id: string | null) => void;
      onFocusSuggestion: (callback: (id: string | null) => void) => () => void;
    };
  }
}
