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

interface SessionState {
  // Current session
  currentSession: Session | null;
  transcript: TranscriptEntry[];
  notes: NoteEntry[];
  participants: Participant[];
  sessionContext: string;

  // AI state
  latestSuggestion: string | null;
  aiStreaming: boolean;
  currentToolName: string | null;
  searchResults: SearchResult[];
  webSearchResults: WebSearchResult[];
  codeAnalysis: CodeAnalysis | null;

  // Agent activity log
  agentLog: AgentLogEntry[];

  // Connection state
  serverOnline: boolean;
  audioConnected: boolean;
  sessionConnected: boolean;
  recording: boolean;
  recordingStartedAt: number | null;

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

  // Actions
  setCurrentSession: (session: Session | null) => void;
  setSessionContext: (context: string) => void;
  addTranscriptEntry: (entry: Omit<TranscriptEntry, "id" | "fact_checks"> & { id?: string }) => void;
  relabelSpeaker: (entryId: string, newSpeaker: string) => void;
  addFactChecks: (transcriptIndex: number, checks: FactCheck[]) => void;
  addNotes: (notes: NoteEntry[]) => void;
  renameSpeaker: (speakerLabel: string, displayName: string) => void;
  updateParticipant: (participant: Participant) => void;
  renameCurrentSession: (name: string) => void;
  deleteTranscriptEntry: (entryId: string) => void;
  editTranscriptEntry: (entryId: string, newText: string) => void;
  setSuggestion: (text: string | null) => void;
  setAIStreaming: (streaming: boolean) => void;
  setServerOnline: (online: boolean) => void;
  setAudioConnected: (connected: boolean) => void;
  setSessionConnected: (connected: boolean) => void;
  setRecording: (recording: boolean) => void;
  setSessions: (sessions: SessionSummary[]) => void;
  handleAIEvent: (event: AIEvent) => void;
  clearAgentLog: () => void;
  setUserSearch: (query: string, results: SearchResult[], loading: boolean) => void;
  clearUserSearch: () => void;
  setInsightsDrawerOpen: (open: boolean) => void;
  setInsightsDrawerTab: (tab: InsightsTab) => void;
  incrementUnseenInsights: () => void;
  clearUnseenInsights: () => void;
  syncFromMain: (data: { transcript: TranscriptEntry[]; latestSuggestion: string | null; notes: NoteEntry[] }) => void;
  reset: () => void;
}

const genId = () => crypto.randomUUID().replace(/-/g, "").slice(0, 12);

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

export const useSessionStore = create<SessionState>()((set, get) => ({
  currentSession: null,
  transcript: [],
  notes: [],
  participants: [],
  sessionContext: "",
  latestSuggestion: null,
  aiStreaming: false,
  currentToolName: null,
  searchResults: [],
  webSearchResults: [],
  codeAnalysis: null,
  agentLog: [],
  serverOnline: false,
  audioConnected: false,
  sessionConnected: false,
  recording: false,
  recordingStartedAt: null,
  sessions: [],
  insightsDrawerOpen: false,
  insightsDrawerTab: "suggestions" as InsightsTab,
  unseenInsightCount: 0,
  userSearchQuery: "",
  userSearchResults: [],
  userSearchLoading: false,

  setCurrentSession: (session) => {
    // Push session settings overrides into the settings store
    useSettingsStore.getState().setSessionOverrides(session?.settings ?? null);

    set({
      currentSession: session,
      transcript: session?.transcript ?? [],
      notes: session?.notes ?? [],
      participants: session?.participants ?? [],
      sessionContext: session?.context ?? "",
      latestSuggestion: null,
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

  addFactChecks: (transcriptIndex, checks) =>
    set((state) => {
      const updated = [...state.transcript];
      if (updated[transcriptIndex]) {
        updated[transcriptIndex] = {
          ...updated[transcriptIndex],
          fact_checks: [...updated[transcriptIndex].fact_checks, ...checks],
        };
      }
      return { transcript: updated };
    }),

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

  setSuggestion: (text) => set({ latestSuggestion: text }),
  setAIStreaming: (streaming) => set({ aiStreaming: streaming }),
  setServerOnline: (online) => set({ serverOnline: online }),
  setAudioConnected: (connected) => set({ audioConnected: connected }),
  setSessionConnected: (connected) => set({ sessionConnected: connected }),
  setRecording: (recording) =>
    set({ recording, recordingStartedAt: recording ? Date.now() : null }),
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
        const updates: Partial<SessionState> = {
          aiStreaming: true,
          latestSuggestion: (state.latestSuggestion || "") + event.text,
        };
        // Add "Thinking..." log entry only on first delta
        if (isFirst) {
          updates.agentLog = [
            ...state.agentLog,
            { id: genId(), timestamp: now, type: "thinking", summary: "Thinking..." },
          ];
        }
        set(updates as SessionState);
        break;
      }

      case "tool_call":
        set({
          currentToolName: event.name,
          agentLog: [
            ...state.agentLog,
            {
              id: genId(),
              timestamp: now,
              type: "tool_call",
              name: event.name,
              summary: `Calling ${event.name}`,
            },
          ],
        });
        break;

      case "tool_result": {
        const summary = summarizeToolResult(event.name, event.result);
        const detail =
          event.name === "deep_think"
            ? (event.result as { reasoning?: string }).reasoning
            : undefined;

        set({
          currentToolName: null,
          agentLog: [
            ...state.agentLog,
            {
              id: genId(),
              timestamp: now,
              type: "tool_result",
              name: event.name,
              summary: `${event.name}: ${summary}`,
              detail,
            },
          ],
        });

        // Existing tool result handling
        if (event.name === "fact_check") {
          const result = event.result as { claims?: FactCheck[] };
          if (result.claims && state.transcript.length > 0) {
            const idx = state.transcript.length - 1;
            state.addFactChecks(idx, result.claims);
          }
        } else if (event.name === "suggest_response") {
          const result = event.result as { suggestion?: string };
          if (result.suggestion) {
            set({ latestSuggestion: result.suggestion });
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
            state.addNotes(noteEntries);
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
          agentLog: [
            ...state.agentLog,
            { id: genId(), timestamp: now, type: "done", summary: "Done" },
          ],
        });
        break;

      case "error":
        set({
          aiStreaming: false,
          currentToolName: null,
          agentLog: [
            ...state.agentLog,
            {
              id: genId(),
              timestamp: now,
              type: "error",
              summary: `Error: ${event.message}`,
            },
          ],
        });
        console.error("AI error:", event.message);
        break;
    }
  },

  syncFromMain: (data) =>
    set({
      transcript: data.transcript,
      latestSuggestion: data.latestSuggestion,
      notes: data.notes,
    }),

  reset: () =>
    set({
      currentSession: null,
      transcript: [],
      notes: [],
      participants: [],
      sessionContext: "",
      latestSuggestion: null,
      aiStreaming: false,
      currentToolName: null,
      searchResults: [],
      webSearchResults: [],
      codeAnalysis: null,
      agentLog: [],
      recording: false,
      recordingStartedAt: null,
      insightsDrawerOpen: false,
      insightsDrawerTab: "suggestions" as InsightsTab,
      unseenInsightCount: 0,
      userSearchQuery: "",
      userSearchResults: [],
      userSearchLoading: false,
    }),
}));
