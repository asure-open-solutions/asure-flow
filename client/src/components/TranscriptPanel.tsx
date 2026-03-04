import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { searchTranscripts, deleteTranscriptEntry, editTranscriptEntry } from "@/services/api";
import { FactCheckBadge } from "./FactCheckBadge";
import { cn } from "@/lib/utils";
import { Mic, Monitor, Users, Search, X, Loader2, Pencil, Check, Copy, Trash2, ShieldX } from "lucide-react";

// ── Dynamic speaker colors ──

const SPEAKER_COLOR_PALETTE = [
  "text-violet-400",
  "text-emerald-400",
  "text-amber-400",
  "text-pink-400",
  "text-cyan-400",
  "text-orange-400",
];

function makeSpeakerColorFn() {
  const cache = new Map<string, string>();
  return function (speaker: string): string {
    if (speaker === "User") return "text-blue-400";
    if (speaker === "Third Party") return "text-violet-400";
    if (!cache.has(speaker)) {
      const idx = cache.size % SPEAKER_COLOR_PALETTE.length;
      cache.set(speaker, SPEAKER_COLOR_PALETTE[idx]);
    }
    return cache.get(speaker)!;
  };
}

function SpeakerIcon({ speaker }: { speaker: string }) {
  if (speaker === "User") return <Mic className="h-3.5 w-3.5 inline mr-1" />;
  if (speaker === "Third Party") return <Monitor className="h-3.5 w-3.5 inline mr-1" />;
  return <Users className="h-3.5 w-3.5 inline mr-1" />;
}

/** Inline speaker rename popover */
function SpeakerRenamePopover({
  speaker,
  onRename,
  onClose,
}: {
  speaker: string;
  onRename: (name: string) => void;
  onClose: () => void;
}) {
  const [value, setValue] = useState(speaker);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  // Close on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (trimmed && trimmed !== speaker) {
      onRename(trimmed);
    }
    onClose();
  };

  return (
    <div ref={containerRef} className="absolute left-0 top-full z-20 mt-1 flex items-center gap-1 rounded-md border border-white/20 bg-zinc-800 px-2 py-1 shadow-lg">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSubmit();
          if (e.key === "Escape") onClose();
        }}
        className="w-28 rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-xs text-white focus:border-blue-500/30 focus:outline-none"
      />
      <button onClick={handleSubmit} className="rounded p-0.5 text-emerald-400 hover:bg-emerald-400/10">
        <Check className="h-3 w-3" />
      </button>
      <button onClick={onClose} className="rounded p-0.5 text-white/40 hover:bg-white/10">
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

export function TranscriptPanel() {
  const transcript = useSessionStore((s) => s.transcript);
  const participants = useSessionStore((s) => s.participants);
  const currentSession = useSessionStore((s) => s.currentSession);
  const userSearchQuery = useSessionStore((s) => s.userSearchQuery);
  const userSearchResults = useSessionStore((s) => s.userSearchResults);
  const userSearchLoading = useSessionStore((s) => s.userSearchLoading);
  const setUserSearch = useSessionStore((s) => s.setUserSearch);
  const clearUserSearch = useSessionStore((s) => s.clearUserSearch);
  const renameSpeaker = useSessionStore((s) => s.renameSpeaker);
  const storeDeleteEntry = useSessionStore((s) => s.deleteTranscriptEntry);
  const storeEditEntry = useSessionStore((s) => s.editTranscriptEntry);
  const requestRerun = useSessionStore((s) => s.requestRerun);
  const clearFactChecks = useSessionStore((s) => s.clearFactChecks);
  const hasFactChecks = useSessionStore((s) => s.transcript.some((e) => e.fact_checks.length > 0));
  const [confirmClearFacts, setConfirmClearFacts] = useState(false);

  // Per-session speaker color function — reset when session changes so colors
  // are assigned from scratch for each session rather than accumulating globally.
  const speakerColor = useMemo(() => makeSpeakerColorFn(), [currentSession?.id]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const [showSearch, setShowSearch] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState<string>("");
  const [renamingSpkr, setRenamingSpkr] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Cleanup pending debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  // Transcript entry editing state
  const [editingEntryId, setEditingEntryId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Resolve display name from participants list
  const displayName = useCallback(
    (speaker: string): string => {
      const p = participants.find((x) => x.speaker_label === speaker);
      return p?.display_name || speaker;
    },
    [participants],
  );

  const handleRenameSpeaker = useCallback(
    (speakerLabel: string, newName: string) => {
      renameSpeaker(speakerLabel, newName);
    },
    [renameSpeaker],
  );

  // Collect unique speakers for filter dropdown
  const speakers = useMemo(() => {
    const set = new Set<string>();
    for (const e of transcript) set.add(e.speaker);
    return Array.from(set).sort();
  }, [transcript]);

  // Matched entry IDs for highlighting
  const matchedIds = useMemo(() => {
    const set = new Set<string>();
    for (const r of userSearchResults) {
      if (r.entry_id) set.add(r.entry_id);
    }
    return set;
  }, [userSearchResults]);

  // Auto-scroll to bottom (only when not searching)
  useEffect(() => {
    if (scrollRef.current && !userSearchQuery) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript, userSearchQuery]);

  const performSearch = useCallback(
    async (query: string, speaker?: string) => {
      if (!query.trim()) {
        clearUserSearch();
        return;
      }
      setUserSearch(query, [], true);
      try {
        const res = await searchTranscripts({
          query,
          session_id: currentSession?.id,
          speaker: speaker || undefined,
          max_results: 20,
        });
        setUserSearch(query, res.results, false);
      } catch {
        setUserSearch(query, [], false);
      }
    },
    [currentSession?.id, setUserSearch, clearUserSearch],
  );

  // Debounced search
  const handleSearchChange = useCallback(
    (value: string) => {
      setSearchInput(value);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        performSearch(value, speakerFilter);
      }, 300);
    },
    [performSearch, speakerFilter],
  );

  const handleSpeakerFilterChange = useCallback(
    (value: string) => {
      setSpeakerFilter(value);
      if (searchInput.trim()) {
        performSearch(searchInput, value);
      }
    },
    [searchInput, performSearch],
  );

  const handleCloseSearch = useCallback(() => {
    setShowSearch(false);
    setSearchInput("");
    setSpeakerFilter("");
    clearUserSearch();
  }, [clearUserSearch]);

  // ── Transcript entry actions ──

  const handleCopy = useCallback(async (entryId: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(entryId);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {
      // Fallback: silently fail
    }
  }, []);

  const startEditing = useCallback((entryId: string, text: string) => {
    setEditingEntryId(entryId);
    setEditValue(text);
  }, []);

  const handleSaveEdit = useCallback(
    async (entryId: string) => {
      const trimmed = editValue.trim();
      if (!trimmed || !currentSession) {
        setEditingEntryId(null);
        return;
      }
      try {
        await editTranscriptEntry(currentSession.id, entryId, trimmed);
        storeEditEntry(entryId, trimmed);
        requestRerun();
      } catch (err) {
        console.error("Failed to edit entry:", err);
      }
      setEditingEntryId(null);
    },
    [editValue, currentSession, storeEditEntry, requestRerun],
  );

  const handleDeleteEntry = useCallback(
    async (entryId: string) => {
      if (!currentSession) return;
      try {
        await deleteTranscriptEntry(currentSession.id, entryId);
        storeDeleteEntry(entryId);
        requestRerun();
      } catch (err) {
        console.error("Failed to delete entry:", err);
      }
    },
    [currentSession, storeDeleteEntry, requestRerun],
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-white/80">
          <Mic className="h-4 w-4" />
          Live Transcript
        </h2>
        <div className="flex items-center gap-1">
          {hasFactChecks && (
            confirmClearFacts ? (
              <span className="flex items-center gap-1 text-xs text-white/50 mr-1">
                Clear checks?
                <button
                  onClick={() => { clearFactChecks(); setConfirmClearFacts(false); }}
                  className="rounded p-0.5 text-red-400 hover:bg-red-400/10 transition-colors"
                  aria-label="Confirm clear fact-checks"
                >
                  <Check className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => setConfirmClearFacts(false)}
                  className="rounded p-0.5 text-white/40 hover:text-white/70 transition-colors"
                  aria-label="Cancel clear"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </span>
            ) : (
              <button
                onClick={() => setConfirmClearFacts(true)}
                className="rounded-md p-1.5 text-white/40 hover:text-white/70 hover:bg-white/5 transition-colors"
                title="Clear all fact-checks"
                aria-label="Clear all fact-checks"
              >
                <ShieldX className="h-4 w-4" />
              </button>
            )
          )}
          <button
            onClick={() => (showSearch ? handleCloseSearch() : setShowSearch(true))}
            className={cn(
              "rounded-md p-1.5 transition-colors",
              showSearch
                ? "text-blue-400 bg-blue-500/10"
                : "text-white/40 hover:text-white/70 hover:bg-white/5",
            )}
            title="Search transcript"
          >
            <Search className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Search bar */}
      {showSearch && (
        <div className="border-b border-white/10 px-4 py-2 space-y-2">
          <div className="flex items-center gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-white/30" />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => handleSearchChange(e.target.value)}
                placeholder="Search transcript..."
                autoFocus
                className="w-full rounded-md border border-white/10 bg-white/[0.03] py-1.5 pl-8 pr-8 text-sm text-white placeholder:text-white/30 focus:border-blue-500/30 focus:outline-none"
              />
              {searchInput && (
                <button
                  onClick={() => handleSearchChange("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            {speakers.length > 1 && (
              <select
                value={speakerFilter}
                onChange={(e) => handleSpeakerFilterChange(e.target.value)}
                className="rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5 text-xs text-white/70 focus:border-blue-500/30 focus:outline-none"
              >
                <option value="">All speakers</option>
                {speakers.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            )}
          </div>
          {userSearchLoading && (
            <div className="flex items-center gap-1.5 text-xs text-white/40">
              <Loader2 className="h-3 w-3 animate-spin" />
              Searching...
            </div>
          )}
          {!userSearchLoading && userSearchQuery && (
            <p className="text-xs text-white/40">
              {userSearchResults.length} result{userSearchResults.length !== 1 ? "s" : ""}
            </p>
          )}
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {transcript.length === 0 && (
          <p className="text-sm text-white/40 text-center py-8">
            Start recording to see the transcript here.
          </p>
        )}

        {transcript.map((entry) => {
          const isMatch = userSearchQuery && matchedIds.has(entry.id);
          return (
            <div
              key={entry.id}
              className={cn(
                "group relative rounded-md px-1 -mx-1 transition-colors",
                isMatch && "bg-blue-500/10 ring-1 ring-blue-500/20",
                userSearchQuery && !isMatch && "opacity-40",
              )}
            >
              {/* Hover action buttons */}
              {editingEntryId !== entry.id && (
                <div className="absolute right-1 top-0 hidden group-hover:flex items-center gap-0.5 z-10">
                  <button
                    onClick={() => handleCopy(entry.id, entry.text)}
                    className="rounded p-1 text-white/30 hover:text-white/70 hover:bg-white/10 transition-colors"
                    title={copiedId === entry.id ? "Copied!" : "Copy text"}
                  >
                    {copiedId === entry.id ? (
                      <Check className="h-3 w-3 text-emerald-400" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </button>
                  <button
                    onClick={() => startEditing(entry.id, entry.text)}
                    className="rounded p-1 text-white/30 hover:text-blue-400 hover:bg-blue-400/10 transition-colors"
                    title="Edit text"
                  >
                    <Pencil className="h-3 w-3" />
                  </button>
                  <button
                    onClick={() => handleDeleteEntry(entry.id)}
                    className="rounded p-1 text-white/30 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                    title="Delete entry"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              )}

              <div className="flex items-start gap-2">
                <span className="relative shrink-0 mt-0.5">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setRenamingSpkr(renamingSpkr === entry.speaker ? null : entry.speaker);
                    }}
                    className={cn(
                      "group/spkr flex items-center gap-0.5 text-xs font-semibold transition-colors",
                      speakerColor(entry.speaker),
                    )}
                    title={`Click to rename ${entry.speaker}`}
                  >
                    <SpeakerIcon speaker={entry.speaker} />
                    {displayName(entry.speaker)}
                    <Pencil className="h-2.5 w-2.5 opacity-0 group-hover/spkr:opacity-50 transition-opacity" />
                  </button>
                  {renamingSpkr === entry.speaker && (
                    <SpeakerRenamePopover
                      speaker={entry.speaker}
                      onRename={(name) => handleRenameSpeaker(entry.speaker, name)}
                      onClose={() => setRenamingSpkr(null)}
                    />
                  )}
                </span>

                {editingEntryId === entry.id ? (
                  <div className="flex-1 min-w-0">
                    <textarea
                      autoFocus
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSaveEdit(entry.id);
                        }
                        if (e.key === "Escape") setEditingEntryId(null);
                      }}
                      className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-sm text-white resize-none focus:border-blue-500/30 focus:outline-none"
                      rows={2}
                    />
                    <div className="flex gap-1 mt-1">
                      <button
                        onClick={() => handleSaveEdit(entry.id)}
                        className="flex items-center gap-1 rounded px-2 py-0.5 text-xs text-emerald-400 hover:bg-emerald-400/10 transition-colors"
                      >
                        <Check className="h-3 w-3" /> Save
                      </button>
                      <button
                        onClick={() => setEditingEntryId(null)}
                        className="flex items-center gap-1 rounded px-2 py-0.5 text-xs text-white/40 hover:bg-white/10 transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-white/90 leading-relaxed">{entry.text}</p>
                )}
              </div>

              {entry.fact_checks.length > 0 && (
                <div className="mt-1.5 ml-6 flex flex-wrap gap-1.5">
                  {entry.fact_checks.map((fc, i) => (
                    <FactCheckBadge key={i} check={fc} />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
