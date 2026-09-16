import { create } from "zustand";
import { useSettingsStore } from "@/stores/settingsStore";
import type {
  TranscriptEntry,
  NoteEntry,
  FactCheck,
  Session,
  SessionSummary,
  AIEvent,
  AgentLogEntry,
  Participant,
  InsightsTab,
} from "@/types";

export interface SearchResult {
  speaker: string;
  text: string;
  timestamp: string;
  session_name?: string;
  session_id?: string;
  entry_id?: string;
  relevance?: number;
}

export interface WebSearchResult {
  title: string;
  snippet: string;
  url: string;
  credibility?: { tier: string; score: number };
}

export interface CodeAnalysis {
  code: string;
  language?: string;
  analysis: string;
}

export interface SuggestionEntry {
  id: string;
  text: string;
  responding_to: string;
  timestamp: string;
}

interface SessionState {
  // Current session
  currentSession: Session | null;
  transcript: TranscriptEntry[];
  notes: NoteEntry[];
  participants: Participant[];
  sessionContext: string;

  // AI state
  suggestions: SuggestionEntry[];
  focusedSuggestionId: string | null;
  // True while the user is delivering a suggestion (server echo-lock). The live
  // card freezes and the AI holds new suggestions until the other party responds.
  suggestionsLocked: boolean;
  aiStreaming: boolean;
  currentToolName: string | null;
  searchResults: SearchResult[];
  webSearchResults: WebSearchResult[];
  codeAnalysis: CodeAnalysis | null;

  // Agent activity log
  agentLog: AgentLogEntry[];

  // Connection state
  serverOnline: boolean;
  llmAvailable: boolean;
  llmProvider: string | null;
  connectionLatencyMs: number | null;
  whisperLoaded: boolean;
  whisperDevice: string | null;
  lastAiLatencyMs: number | null;
  audioConnected: boolean;
  sessionConnected: boolean;
  recording: boolean;
  recordingStartedAt: number | null;
  audioWarning: string | null;
  // Whisper accuracy-degradation notice (e.g. OOM fell back to a weaker model).
  transcriptionWarning: string | null;

  // Overlay-synced audio toggles (read-only in overlay)
  overlayAudioToggles: { mic: boolean; system: boolean };

  // Session list
  sessions: SessionSummary[];

  // Insights drawer
  insightsDrawerOpen: boolean;
  insightsDrawerTab: InsightsTab;
  unseenInsightCount: number;

  // User search
  userSearchQuery: string;
  userSearchResults: SearchResult[];
  userSearchLoading: boolean;

  // Agent rerun flag (set by components, consumed by App.tsx to send WS message)
  rerunRequested: boolean;

  // Actions
  setCurrentSession: (session: Session | null) => void;
  setSessionContext: (context: string) => void;
  addTranscriptEntry: (entry: Omit<TranscriptEntry, "id" | "fact_checks"> & { id?: string }) => void;
  relabelSpeaker: (entryId: string, newSpeaker: string) => void;
  addFactChecks: (transcriptId: string, checks: FactCheck[]) => void;
  addNotes: (notes: NoteEntry[]) => void;
  renameSpeaker: (speakerLabel: string, displayName: string) => void;
  updateParticipant: (participant: Participant) => void;
  renameCurrentSession: (name: string) => void;
  deleteTranscriptEntry: (entryId: string) => void;
  editTranscriptEntry: (entryId: string, newText: string) => void;
  addSuggestion: (text: string, respondingTo?: string) => void;
  focusSuggestion: (id: string | null) => void;
  setAIStreaming: (streaming: boolean) => void;
  setServerOnline: (online: boolean) => void;
  setLlmStatus: (available: boolean, provider: string | null) => void;
  setConnectionDiagnostics: (latencyMs: number | null, whisperLoaded: boolean, whisperDevice: string | null) => void;
  setAudioConnected: (connected: boolean) => void;
  setSessionConnected: (connected: boolean) => void;
  setRecording: (recording: boolean) => void;
  setAudioWarning: (warning: string | null) => void;
  setTranscriptionWarning: (warning: string | null) => void;
  setSessions: (sessions: SessionSummary[]) => void;
  handleAIEvent: (event: AIEvent) => void;
  clearAgentLog: () => void;
  setUserSearch: (query: string, results: SearchResult[], loading: boolean) => void;
  clearUserSearch: () => void;
  setInsightsDrawerOpen: (open: boolean) => void;
  setInsightsDrawerTab: (tab: InsightsTab) => void;
  incrementUnseenInsights: () => void;
  clearUnseenInsights: () => void;
  syncFromMain: (data: {
    transcript: TranscriptEntry[];
    latestSuggestion: string | null;
    focusedSuggestionId?: string | null;
    focusedSuggestionText?: string | null;
    notes: NoteEntry[];
    recording?: boolean;
    recordingStartedAt?: number | null;
    audioToggles?: { mic: boolean; system: boolean };
  }) => void;
  clearSuggestions: () => void;
  clearNotes: () => void;
  toggleNoteCompleted: (noteId: string) => void;
  clearFactChecks: () => void;
  requestRerun: () => void;
  clearRerunRequest: () => void;
  reset: () => void;
}

// crypto.randomUUID exists only in secure contexts (https/localhost). Remote
// clients connecting over plain http:// would otherwise crash on the first
// transcript line, so fall back to a non-crypto id there.
export const genId = (): string => {
  try {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
    }
  } catch {
    // fall through
  }
  return (Date.now().toString(36) + Math.random().toString(36).slice(2, 10)).slice(0, 12);
};

/** Build a brief summary string for a tool_result event. */
function summarizeToolResult(name: string, result: Record<string, unknown>): string {
  if (name === "fact_check") {
    const claims = result.claims as unknown[] | undefined;
    return `${claims?.length ?? 0} claim(s) checked`;
  }
  if (name === "suggest_response") return "suggestion generated";
  if (name === "extract_notes") {
    const items = (result.action_items as unknown[] | undefined)?.length ?? 0;
    const decisions = (result.decisions as unknown[] | undefined)?.length ?? 0;
    const facts = (result.key_facts as unknown[] | undefined)?.length ?? 0;
    const risks = (result.risks as unknown[] | undefined)?.length ?? 0;
    return `${items + decisions + facts + risks} note(s) extracted`;
  }
  if (name === "search_transcript" || name === "search_sessions") {
    const count = (result.results as unknown[] | undefined)?.length ?? 0;
    return `${count} result(s)`;
  }
  if (name === "web_search") {
    const count = (result.results as unknown[] | undefined)?.length ?? 0;
    return `${count} result(s)`;
  }
  if (name === "deep_think") {
    const conclusion = (result.conclusion as string) ?? "";
    return conclusion.length > 80 ? conclusion.slice(0, 80) + "..." : conclusion;
  }
  if (name === "format_code") return "code analysed";
  return "completed";
}

const MAX_AGENT_LOG_ENTRIES = 200;

function appendAgentLog(log: AgentLogEntry[], entry: AgentLogEntry): AgentLogEntry[] {
  return [...log, entry].slice(-MAX_AGENT_LOG_ENTRIES);
}

export const useSessionStore = create<SessionState>()((set, get) => ({
  currentSession: null,
  transcript: [],
  notes: [],
  participants: [],
  sessionContext: "",
  suggestions: [],
  focusedSuggestionId: null,
  suggestionsLocked: false,
  aiStreaming: false,
  currentToolName: null,
  searchResults: [],
  webSearchResults: [],
  codeAnalysis: null,
  agentLog: [],
  serverOnline: false,
  llmAvailable: false,
  llmProvider: null,
  connectionLatencyMs: null,
  whisperLoaded: false,
  whisperDevice: null,
  lastAiLatencyMs: null,
  audioConnected: false,
  sessionConnected: false,
  recording: false,
  recordingStartedAt: null,
  audioWarning: null,
  transcriptionWarning: null,
  overlayAudioToggles: { mic: true, system: true },
  sessions: [],
  insightsDrawerOpen: false,
  insightsDrawerTab: "suggestions" as InsightsTab,
  unseenInsightCount: 0,
  userSearchQuery: "",
  userSearchResults: [],
  userSearchLoading: false,
  rerunRequested: false,

  setCurrentSession: (session) => {
    // Push session settings overrides into the settings store
    useSettingsStore.getState().setSessionOverrides(session?.settings ?? null);

    set({
      currentSession: session,
      transcript: session?.transcript ?? [],
      notes: session?.notes ?? [],
      participants: session?.participants ?? [],
      sessionContext: session?.context ?? "",
      suggestions: (session?.suggestions ?? []).map((s) => ({
        id: s.id,
        text: s.text,
        responding_to: s.responding_to ?? "",
        timestamp: s.timestamp,
      })),
      agentLog: [],
    });
  },

  setSessionContext: (context) => set({ sessionContext: context }),

  addTranscriptEntry: (entry) =>
    set((state) => ({
      transcript: [
        ...state.transcript,
        {
          id: entry.id || genId(),
          timestamp: entry.timestamp || new Date().toISOString(),
          speaker: entry.speaker,
          text: entry.text,
          fact_checks: [],
          audio_start: entry.audio_start,
          audio_end: entry.audio_end,
        },
      ],
    })),

  relabelSpeaker: (entryId, newSpeaker) =>
    set((state) => ({
      transcript: state.transcript.map((e) => {
        // Match by audio timing key (format: "start-end")
        const audioKey =
          e.audio_start != null && e.audio_end != null
            ? `${e.audio_start.toFixed(3)}-${e.audio_end.toFixed(3)}`
            : null;
        if (e.id === entryId || audioKey === entryId) {
          return { ...e, speaker: newSpeaker };
        }
        return e;
      }),
    })),

  addFactChecks: (transcriptId, checks) =>
    set((state) => ({
      transcript: state.transcript.map((entry) =>
        entry.id === transcriptId
          ? { ...entry, fact_checks: [...entry.fact_checks, ...checks] }
          : entry,
      ),
    })),

  addNotes: (newNotes) =>
    set((state) => ({
      notes: [...state.notes, ...newNotes],
    })),

  renameSpeaker: (speakerLabel, displayName) =>
    set((state) => ({
      transcript: state.transcript.map((e) =>
        e.speaker === speakerLabel ? { ...e, speaker: displayName } : e,
      ),
    })),

  updateParticipant: (participant) =>
    set((state) => {
      const exists = state.participants.some(
        (p) => p.speaker_label === participant.speaker_label,
      );
      return {
        participants: exists
          ? state.participants.map((p) =>
              p.speaker_label === participant.speaker_label ? participant : p,
            )
          : [...state.participants, participant],
      };
    }),

  renameCurrentSession: (name) =>
    set((state) => ({
      currentSession: state.currentSession
        ? { ...state.currentSession, name }
        : null,
      sessions: state.sessions.map((s) =>
        s.id === state.currentSession?.id ? { ...s, name } : s,
      ),
    })),

  deleteTranscriptEntry: (entryId) =>
    set((state) => ({
      transcript: state.transcript.filter((e) => e.id !== entryId),
    })),

  editTranscriptEntry: (entryId, newText) =>
    set((state) => ({
      transcript: state.transcript.map((e) =>
        e.id === entryId ? { ...e, text: newText } : e,
      ),
    })),

  addSuggestion: (text, respondingTo = "") =>
    set((state) => ({
      suggestions: [
        ...state.suggestions,
        { id: genId(), text, responding_to: respondingTo, timestamp: new Date().toISOString() },
      ],
    })),
  focusSuggestion: (id) =>
    set((state) => ({
      focusedSuggestionId: state.focusedSuggestionId === id ? null : id,
    })),
  clearSuggestions: () => set({ suggestions: [], focusedSuggestionId: null, suggestionsLocked: false }),
  clearNotes: () => set({ notes: [] }),
  toggleNoteCompleted: (noteId) =>
    set((state) => ({
      notes: state.notes.map((n) =>
        n.id === noteId ? { ...n, completed: !n.completed } : n,
      ),
    })),
  clearFactChecks: () =>
    set((state) => ({
      transcript: state.transcript.map((e) =>
        e.fact_checks.length > 0 ? { ...e, fact_checks: [] } : e,
      ),
    })),
  requestRerun: () => set({ rerunRequested: true }),
  clearRerunRequest: () => set({ rerunRequested: false }),
  setAIStreaming: (streaming) => set({ aiStreaming: streaming }),
  setServerOnline: (online) => set({ serverOnline: online }),
  setLlmStatus: (available, provider) => set({ llmAvailable: available, llmProvider: provider }),
  setConnectionDiagnostics: (connectionLatencyMs, whisperLoaded, whisperDevice) =>
    set({ connectionLatencyMs, whisperLoaded, whisperDevice }),
  setAudioConnected: (connected) => set({ audioConnected: connected }),
  setSessionConnected: (connected) => set({ sessionConnected: connected }),
  setRecording: (recording) =>
    set({ recording, recordingStartedAt: recording ? Date.now() : null, ...(!recording && { audioWarning: null }) }),
  setAudioWarning: (warning) => set({ audioWarning: warning }),
  setTranscriptionWarning: (warning) => set({ transcriptionWarning: warning }),
  setSessions: (sessions) => set({ sessions }),
  clearAgentLog: () => set({ agentLog: [] }),

  setUserSearch: (query, results, loading) =>
    set({ userSearchQuery: query, userSearchResults: results, userSearchLoading: loading }),

  clearUserSearch: () =>
    set({ userSearchQuery: "", userSearchResults: [], userSearchLoading: false }),

  setInsightsDrawerOpen: (open) => set({ insightsDrawerOpen: open, ...(open ? { unseenInsightCount: 0 } : {}) }),
  setInsightsDrawerTab: (tab) => set({ insightsDrawerTab: tab }),
  incrementUnseenInsights: () => set((state) => ({ unseenInsightCount: state.insightsDrawerOpen ? 0 : state.unseenInsightCount + 1 })),
  clearUnseenInsights: () => set({ unseenInsightCount: 0 }),

  handleAIEvent: (event) => {
    const state = get();
    const now = new Date().toISOString();

    switch (event.type) {
      case "content_delta": {
        const isFirst = !state.aiStreaming;
        // Do NOT write content_delta text to latestSuggestion — these deltas contain
        // raw LLM output which may include function-call XML from providers that stream
        // tool calls as text. The real suggestion arrives via tool_result.
        if (isFirst) {
          set({
            aiStreaming: true,
            agentLog: appendAgentLog(state.agentLog, {
              id: genId(), timestamp: now, type: "thinking", summary: "Thinking...",
            }),
          });
        }
        break;
      }

      case "tool_call":
        set({
          currentToolName: event.name,
          agentLog: appendAgentLog(state.agentLog, {
            id: genId(),
            timestamp: now,
            type: "tool_call",
            name: event.name,
            summary: `Calling ${event.name}`,
            specialist: event.specialist,
          }),
        });
        break;

      case "tool_result": {
        if (event.latency_ms != null) {
          set({ lastAiLatencyMs: event.latency_ms });
        }
        const summary = summarizeToolResult(event.name, event.result);
        const detail =
          event.name === "deep_think"
            ? (event.result as { reasoning?: string }).reasoning
            : undefined;

        set({
          currentToolName: null,
          agentLog: appendAgentLog(state.agentLog, {
            id: genId(),
            timestamp: now,
            type: "tool_result",
            name: event.name,
            summary: event.specialist
              ? `[${event.specialist}] ${event.name}: ${summary}`
              : `${event.name}: ${summary}`,
            detail,
            specialist: event.specialist,
          }),
        });

        // Existing tool result handling
        if (event.name === "fact_check") {
          const result = event.result as { claims?: FactCheck[] };
          if (result.claims && result.claims.length > 0) {
            const freshTranscript = get().transcript;
            const targetId = event.transcript_id ?? freshTranscript[freshTranscript.length - 1]?.id;
            if (targetId) {
              get().addFactChecks(targetId, result.claims);
            }
          }
        } else if (event.name === "suggest_response") {
          const result = event.result as { suggestion?: string; responding_to?: string };
          if (result.suggestion) {
            get().addSuggestion(result.suggestion, result.responding_to);
            get().incrementUnseenInsights();
          }
        } else if (event.name === "extract_notes") {
          const result = event.result as Record<string, unknown[]>;
          const noteEntries: NoteEntry[] = [];
          // Handle action_items: may be strings or {content, owner, due_date}
          for (const item of (result.action_items ?? []) as unknown[]) {
            if (typeof item === "string" && item) {
              noteEntries.push({
                id: genId(),
                type: "action_item",
                content: item,
                timestamp: new Date().toISOString(),
              });
            } else if (typeof item === "object" && item !== null) {
              const obj = item as Record<string, string>;
              if (obj.content) {
                noteEntries.push({
                  id: genId(),
                  type: "action_item",
                  content: obj.content,
                  owner: obj.owner,
                  due_date: obj.due_date,
                  timestamp: new Date().toISOString(),
                });
              }
            }
          }
          // decisions, key_facts, risks remain as string arrays
          const typeMap: Record<string, NoteEntry["type"]> = {
            decisions: "decision",
            key_facts: "key_fact",
            risks: "risk",
          };
          for (const [key, type] of Object.entries(typeMap)) {
            for (const item of (result[key] ?? []) as string[]) {
              if (item) {
                noteEntries.push({
                  id: genId(),
                  type,
                  content: item,
                  timestamp: new Date().toISOString(),
                });
              }
            }
          }
          if (noteEntries.length > 0) {
            get().addNotes(noteEntries);
            get().incrementUnseenInsights();
          }
        } else if (event.name === "search_transcript" || event.name === "search_sessions") {
          const result = event.result as { results?: SearchResult[] };
          if (result.results) {
            set({ searchResults: result.results });
          }
        } else if (event.name === "web_search") {
          const result = event.result as { results?: WebSearchResult[] };
          if (result.results) {
            set({ webSearchResults: result.results });
          }
        } else if (event.name === "format_code") {
          const result = event.result as unknown as CodeAnalysis;
          if (result.code) {
            set({ codeAnalysis: result });
          }
        }
        break;
      }

      case "done":
        set({
          aiStreaming: false,
          currentToolName: null,
          agentLog: appendAgentLog(state.agentLog, {
            id: genId(),
            timestamp: now,
            type: "done",
            summary: event.reason === "preempted" ? "Preempted — new input" : "Done",
          }),
        });
        break;

      case "preempted":
        // Agent was cancelled because new speech arrived — clear in-progress state
        set({
          aiStreaming: false,
          currentToolName: null,
          agentLog: appendAgentLog(state.agentLog, {
            id: genId(), timestamp: now, type: "done", summary: "Preempted — new input",
          }),
        });
        break;

      case "suggestion_lock":
        // The user is delivering a suggestion (or has stopped). Freeze/unfreeze the
        // live card. On lock, auto-pin the current focal suggestion so it stays put
        // even if a queued suggestion arrives — the automatic replacement for manual pin.
        set({
          suggestionsLocked: event.locked,
          focusedSuggestionId:
            event.locked && !state.focusedSuggestionId && state.suggestions.length > 0
              ? state.suggestions[state.suggestions.length - 1].id
              : state.focusedSuggestionId,
        });
        break;

      case "error":
        set({
          aiStreaming: false,
          currentToolName: null,
          agentLog: appendAgentLog(state.agentLog, {
            id: genId(),
            timestamp: now,
            type: "error",
            summary: `Error: ${event.message}`,
          }),
        });
        console.error("AI error:", event.message);
        break;
    }
  },

  syncFromMain: (data) => {
    const suggestions: SuggestionEntry[] = [];
    if (data.focusedSuggestionText) {
      suggestions.push({ id: data.focusedSuggestionId ?? "focused", text: data.focusedSuggestionText, responding_to: "", timestamp: new Date().toISOString() });
    }
    if (data.latestSuggestion && data.latestSuggestion !== data.focusedSuggestionText) {
      suggestions.push({ id: "overlay", text: data.latestSuggestion, responding_to: "", timestamp: new Date().toISOString() });
    }
    if (!suggestions.length && data.latestSuggestion) {
      suggestions.push({ id: "overlay", text: data.latestSuggestion, responding_to: "", timestamp: new Date().toISOString() });
    }
    set({
      transcript: data.transcript,
      suggestions,
      focusedSuggestionId: data.focusedSuggestionId ?? null,
      notes: data.notes,
      recording: data.recording ?? false,
      recordingStartedAt: data.recordingStartedAt ?? null,
      overlayAudioToggles: data.audioToggles ?? { mic: true, system: true },
    });
  },

  reset: () =>
    set({
      currentSession: null,
      transcript: [],
      notes: [],
      participants: [],
      sessionContext: "",
      suggestions: [],
      focusedSuggestionId: null,
      suggestionsLocked: false,
      aiStreaming: false,
      currentToolName: null,
      searchResults: [],
      webSearchResults: [],
      codeAnalysis: null,
      agentLog: [],
      recording: false,
      recordingStartedAt: null,
      audioWarning: null,
      transcriptionWarning: null,
      lastAiLatencyMs: null,
      insightsDrawerOpen: false,
      insightsDrawerTab: "suggestions" as InsightsTab,
      unseenInsightCount: 0,
      userSearchQuery: "",
      userSearchResults: [],
      userSearchLoading: false,
      rerunRequested: false,
    }),
}));
