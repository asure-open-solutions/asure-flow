import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  AppSettings,
  AudioToggles,
  FeatureToggles,
  OverlaySettings,
  SessionSettings,
} from "@/types";

const DEFAULT_SETTINGS: AppSettings = {
  serverUrl: "http://localhost:8000",
  featureToggles: {
    fact_checking: true,
    suggestions: true,
    notes: true,
    search_transcript: true,
    search_sessions: false,
    web_search: true,
    format_code: false,
    deep_think: "off" as const,
  },
  diarization: false,
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
  piiRedaction: false,
  privacyMode: false,
};

interface SettingsState extends AppSettings {
  // Session-level overrides (not persisted in localStorage — loaded from session)
  sessionOverrides: SessionSettings | null;

  setServerUrl: (url: string) => void;
  setFeatureToggles: (toggles: Partial<FeatureToggles>) => void;
  setDiarization: (enabled: boolean) => void;
  setAudioToggles: (toggles: Partial<AudioToggles>) => void;
  setOverlaySettings: (settings: Partial<OverlaySettings>) => void;
  setPiiRedaction: (enabled: boolean) => void;
  setPrivacyMode: (enabled: boolean) => void;
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
      // Don't persist sessionOverrides — they come from the loaded session
      partialize: (state) => {
        const { sessionOverrides, ...rest } = state;
        return rest;
      },
    },
  ),
);
