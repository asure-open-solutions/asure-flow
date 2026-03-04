import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  AppSettings,
  AudioToggles,
  FeatureToggles,
  OverlaySettings,
  ServerConfig,
  SessionSettings,
  UserProfile,
} from "@/types";

/**
 * Detect server URL from environment.
 *
 * Priority:
 *   1. Electron env via preload: process.env.ASUREFLOW_SERVER
 *   2. Vite build-time env: import.meta.env.VITE_ASUREFLOW_SERVER
 *   3. null → fall through to persisted / default value
 */
function getEnvServerUrl(): string | null {
  try {
    // Electron preload exposes this
    const fromElectron = (window as any).electronAPI?.getEnvServerUrl?.();
    if (fromElectron) return fromElectron.replace(/\/+$/, "");
  } catch { /* not in Electron */ }
  try {
    const fromVite = (import.meta as any).env?.VITE_ASUREFLOW_SERVER;
    if (fromVite) return String(fromVite).replace(/\/+$/, "");
  } catch { /* not available */ }
  return null;
}

const DEFAULT_SETTINGS: AppSettings = {
  // Tier 3: device-local
  serverUrl: "http://localhost:8000",
  audioToggles: { mic: true, system: true },
  overlaySettings: {
    contentProtection: true,
    showTranscript: true,
    showFactChecks: true,
    showSuggestions: true,
    showNotes: true,
    overlayMode: "topbar" as const,
    cardPositions: {},
  },
  micDeviceId: null,
  systemDeviceId: null,
  // Tier 2: user profile (defaults used before server profile is fetched)
  featureToggles: {
    fact_checking: true,
    suggestions: true,
    notes: true,
    search_transcript: true,
    search_sessions: false,
    web_search: true,
    format_code: false,
    deep_think: "off" as const,
    agent_mode: "unified" as const,
    parallel_tools: false,
  },
  diarization: false,
  piiRedaction: false,
  privacyMode: false,
  aiPreset: "general",
  customSystemPrompt: null,
};

interface SettingsState extends AppSettings {
  // Session-level overrides (not persisted in localStorage — loaded from session)
  sessionOverrides: SessionSettings | null;

  // Hydration flag — true once persist middleware has restored values from localStorage
  _hydrated: boolean;

  setServerUrl: (url: string) => void;
  setFeatureToggles: (toggles: Partial<FeatureToggles>) => void;
  setDiarization: (enabled: boolean) => void;
  setAudioToggles: (toggles: Partial<AudioToggles>) => void;
  setOverlaySettings: (settings: Partial<OverlaySettings>) => void;
  setPiiRedaction: (enabled: boolean) => void;
  setPrivacyMode: (enabled: boolean) => void;
  setAiPreset: (preset: string) => void;
  setCustomSystemPrompt: (prompt: string | null) => void;
  setMicDeviceId: (id: string | null) => void;
  setSystemDeviceId: (id: string | null) => void;
  /** Sync server-admin values (locked_settings) from a fetched ServerConfig. */
  initFromServerConfig: (config: ServerConfig) => void;
  /** Sync user profile (feature toggles, AI preset, privacy prefs) from server profile. */
  initFromServerProfile: (profile: UserProfile) => void;
  resetAll: () => void;

  // Session overrides management
  setSessionOverrides: (overrides: SessionSettings | null) => void;
  updateSessionOverride: <K extends keyof SessionSettings>(key: K, value: SessionSettings[K]) => void;
  removeSessionOverride: (key: keyof SessionSettings) => void;
  clearSessionOverrides: () => void;

  // Effective values (global merged with session overrides)
  getEffectiveToggles: () => FeatureToggles;
  getEffectiveDiarization: () => boolean;
  getEffectivePiiRedaction: () => boolean;
  getEffectivePrivacyMode: () => boolean;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      ...DEFAULT_SETTINGS,
      sessionOverrides: null,
      _hydrated: false,

      setServerUrl: (url) => set({ serverUrl: url }),

      setFeatureToggles: (toggles) =>
        set((state) => ({
          featureToggles: { ...state.featureToggles, ...toggles },
        })),

      setDiarization: (enabled) => set({ diarization: enabled }),

      setAudioToggles: (toggles) =>
        set((state) => ({
          audioToggles: { ...state.audioToggles, ...toggles },
        })),

      setOverlaySettings: (settings) =>
        set((state) => ({
          overlaySettings: { ...state.overlaySettings, ...settings },
        })),

      setPiiRedaction: (enabled) => set({ piiRedaction: enabled }),

      setPrivacyMode: (enabled) =>
        set((state) => ({
          privacyMode: enabled,
          // Privacy mode force-enables PII redaction and disables web search
          ...(enabled
            ? {
                piiRedaction: true,
                featureToggles: { ...state.featureToggles, web_search: false },
              }
            : {}),
        })),

      setAiPreset: (preset) => set({ aiPreset: preset }),

      setCustomSystemPrompt: (prompt) => set({ customSystemPrompt: prompt }),

      setMicDeviceId: (id) => set({ micDeviceId: id }),

      setSystemDeviceId: (id) => set({ systemDeviceId: id }),

      initFromServerConfig: (_config) => {
        // ServerConfig no longer carries user-preference fields — those come via
        // initFromServerProfile.  This hook remains for future server-admin fields
        // that need client-side reflection (e.g. locked_settings UI feedback).
      },

      initFromServerProfile: (serverProfile) =>
        set((state) => {
          const privacyModeActivated = serverProfile.privacy_mode && !state.privacyMode;
          return {
            featureToggles: {
              fact_checking: serverProfile.fact_checking,
              suggestions: serverProfile.suggestions,
              notes: serverProfile.notes,
              search_transcript: serverProfile.search_transcript,
              search_sessions: serverProfile.search_sessions,
              web_search: privacyModeActivated ? false : serverProfile.web_search,
              format_code: serverProfile.format_code,
              deep_think: serverProfile.deep_think,
              agent_mode: serverProfile.agent_mode ?? "unified",
              parallel_tools: serverProfile.parallel_tools ?? false,
            },
            diarization: serverProfile.diarization_enabled,
            piiRedaction: serverProfile.pii_redaction,
            privacyMode: serverProfile.privacy_mode,
            aiPreset: serverProfile.ai_preset,
            customSystemPrompt: serverProfile.custom_system_prompt,
          };
        }),

      resetAll: () => set({ ...DEFAULT_SETTINGS, sessionOverrides: null }),

      // Session overrides
      setSessionOverrides: (overrides) => set({ sessionOverrides: overrides ?? null }),

      updateSessionOverride: (key, value) =>
        set((state) => ({
          sessionOverrides: { ...state.sessionOverrides, [key]: value },
        })),

      removeSessionOverride: (key) =>
        set((state) => {
          if (!state.sessionOverrides) return {};
          const updated = { ...state.sessionOverrides };
          delete updated[key];
          // If all overrides removed, clear entirely
          const hasAny = Object.values(updated).some((v) => v !== undefined);
          return { sessionOverrides: hasAny ? updated : null };
        }),

      clearSessionOverrides: () => set({ sessionOverrides: null }),

      // Effective values (session overrides take precedence)
      getEffectiveToggles: () => {
        const state = get();
        const o = state.sessionOverrides;
        if (!o) return state.featureToggles;
        return {
          fact_checking: o.fact_checking ?? state.featureToggles.fact_checking,
          suggestions: o.suggestions ?? state.featureToggles.suggestions,
          notes: o.notes ?? state.featureToggles.notes,
          search_transcript: o.search_transcript ?? state.featureToggles.search_transcript,
          search_sessions: o.search_sessions ?? state.featureToggles.search_sessions,
          web_search: o.web_search ?? state.featureToggles.web_search,
          format_code: o.format_code ?? state.featureToggles.format_code,
          deep_think: o.deep_think ?? state.featureToggles.deep_think,
          agent_mode: o.agent_mode ?? state.featureToggles.agent_mode,
          parallel_tools: o.parallel_tools ?? state.featureToggles.parallel_tools,
        };
      },

      getEffectiveDiarization: () => {
        const state = get();
        return state.sessionOverrides?.diarization ?? state.diarization;
      },

      getEffectivePiiRedaction: () => {
        const state = get();
        return state.sessionOverrides?.piiRedaction ?? state.piiRedaction;
      },

      getEffectivePrivacyMode: () => {
        const state = get();
        return state.sessionOverrides?.privacyMode ?? state.privacyMode;
      },
    }),
    {
      name: "asure-flow-settings",
      // Don't persist sessionOverrides — they come from the loaded session.
      // Profile fields (featureToggles, diarization, piiRedaction, privacyMode, aiPreset,
      // customSystemPrompt) are cached in localStorage for fast startup but are always
      // overridden by initFromServerProfile() once the server responds.
      partialize: (state) => {
        const { sessionOverrides, _hydrated, ...rest } = state;
        return rest;
      },
      onRehydrateStorage: () => (state) => {
        if (state) state._hydrated = true;
      },
      // Deep-merge stored state so new nested keys (e.g. overlaySettings.overlayMode)
      // are preserved from the default when an old persisted schema lacks them.
      // Also: ASUREFLOW_SERVER env var always wins when set (for remote-client use).
      merge: (persisted, current) => {
        const envUrl = getEnvServerUrl();
        return {
          ...current,
          ...(persisted as object),
          // Env var takes precedence over persisted URL when explicitly set
          ...(envUrl ? { serverUrl: envUrl } : {}),
          // Deep-merge nested objects so new keys from defaults are preserved
          overlaySettings: {
            ...(current as SettingsState).overlaySettings,
            ...((persisted as SettingsState).overlaySettings ?? {}),
          },
          featureToggles: {
            ...(current as SettingsState).featureToggles,
            ...((persisted as SettingsState).featureToggles ?? {}),
          },
          audioToggles: {
            ...(current as SettingsState).audioToggles,
            ...((persisted as SettingsState).audioToggles ?? {}),
          },
        };
      },
    },
  ),
);
