import { useSessionStore } from "@/stores/sessionStore";
import type { SuggestionEntry } from "@/stores/sessionStore";
import { PanelHeader } from "./PanelHeader";
import { cn } from "@/lib/utils";
import {
  MessageSquare,
  Copy,
  Check,
  Loader2,
  CornerDownLeft,
  Pin,
  PinOff,
  Lock,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";

export function AISuggestionsPanel({ showHeader = true }: { showHeader?: boolean }) {
  const suggestions = useSessionStore((s) => s.suggestions);
  const streaming = useSessionStore((s) => s.aiStreaming);
  const locked = useSessionStore((s) => s.suggestionsLocked);
  const clearSuggestions = useSessionStore((s) => s.clearSuggestions);
  const focusedSuggestionId = useSessionStore((s) => s.focusedSuggestionId);
  const focusSuggestion = useSessionStore((s) => s.focusSuggestion);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  // The "live" card is the pinned suggestion if one is focused, otherwise the newest.
  // Everything else collapses into history so new suggestions never steal the focal slot.
  const focused = suggestions.find((s) => s.id === focusedSuggestionId) ?? null;
  const live: SuggestionEntry | null = focused ?? suggestions[suggestions.length - 1] ?? null;
  const history = suggestions.filter((s) => s.id !== live?.id).reverse();
  const isPinned = focused !== null && focused.id === live?.id;

  const handleCopy = async (entry: SuggestionEntry) => {
    try {
      await navigator.clipboard.writeText(entry.text);
      setCopiedId(entry.id);
      setTimeout(() => setCopiedId(null), 2000);
      // Copying means you're about to deliver it — pin it so it stays the live card.
      if (focusedSuggestionId !== entry.id) focusSuggestion(entry.id);
    } catch {
      // Clipboard access can fail if window is not focused
    }
  };

  const statusLabel = locked ? "Delivering" : isPinned ? "Pinned" : "Latest";

  return (
    <div className="flex h-full flex-col">
      {showHeader && (
        <PanelHeader
          icon={MessageSquare}
          title="Suggestions"
          count={suggestions.length}
          onClear={clearSuggestions}
          clearAriaLabel="Clear all suggestions"
          extra={
            locked ? (
              <span className="flex items-center gap-1 text-[10px] font-medium text-amber-400/90">
                <Lock className="h-3 w-3" /> Locked
              </span>
            ) : streaming ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />
            ) : undefined
          }
        />
      )}

      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
        {suggestions.length === 0 ? (
          <p className="text-sm text-white/30 text-center py-8">
            Response suggestions will appear here during the conversation.
          </p>
        ) : (
          <>
            {/* ── Live card ── */}
            {live && (
              <div
                className={cn(
                  "relative rounded-xl px-4 py-3 transition-colors",
                  locked
                    ? "bg-amber-500/[0.07] border border-amber-400/40 border-l-2 border-l-amber-400"
                    : "bg-blue-500/[0.07] border border-blue-400/40 border-l-2 border-l-blue-400",
                )}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span
                    className={cn(
                      "flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide",
                      locked ? "text-amber-400/90" : "text-blue-400/80",
                    )}
                  >
                    {locked && <Lock className="h-3 w-3" />}
                    {statusLabel}
                  </span>
                  <div className="flex items-center gap-0.5">
                    <button
                      onClick={() => focusSuggestion(live.id)}
                      className={cn(
                        "rounded-md p-1.5 transition-all",
                        isPinned
                          ? "text-blue-400 bg-blue-400/10"
                          : "text-white/40 hover:text-blue-400 hover:bg-blue-400/10",
                      )}
                      title={isPinned ? "Unpin (follow latest)" : "Pin this suggestion"}
                    >
                      {isPinned ? <Pin className="h-3.5 w-3.5" /> : <PinOff className="h-3.5 w-3.5" />}
                    </button>
                    <button
                      onClick={() => handleCopy(live)}
                      className="rounded-md p-1.5 text-white/40 hover:text-white/80 hover:bg-white/10 transition-all"
                      title="Copy to clipboard"
                    >
                      {copiedId === live.id ? (
                        <Check className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </button>
                  </div>
                </div>

                {live.responding_to && (
                  <p className="flex items-start gap-1.5 text-xs text-white/35 mb-1.5 leading-snug">
                    <CornerDownLeft className="h-3 w-3 mt-0.5 shrink-0" />
                    <span className="italic">{live.responding_to}</span>
                  </p>
                )}
                <p className="text-[15px] text-white/95 leading-relaxed whitespace-pre-wrap">
                  {live.text}
                </p>
                <div className="mt-1.5 text-[10px] text-white/25">
                  {locked
                    ? "Holding — new suggestions paused while you respond"
                    : new Date(live.timestamp).toLocaleTimeString()}
                </div>
              </div>
            )}

            {/* ── Collapsible history ── */}
            {history.length > 0 && (
              <div>
                <button
                  onClick={() => setHistoryOpen((o) => !o)}
                  className="flex items-center gap-1 text-xs text-white/40 hover:text-white/70 transition-colors py-1"
                >
                  {historyOpen ? (
                    <ChevronDown className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5" />
                  )}
                  Earlier suggestions ({history.length})
                </button>

                {historyOpen && (
                  <div className="space-y-2 mt-1">
                    {history.map((entry) => (
                      <div
                        key={entry.id}
                        className="group relative rounded-lg px-3 py-2 bg-white/[0.02] border border-white/[0.05] hover:border-white/10 transition-colors"
                      >
                        {entry.responding_to && (
                          <p className="flex items-start gap-1.5 text-[11px] text-white/30 mb-1 leading-snug">
                            <CornerDownLeft className="h-2.5 w-2.5 mt-0.5 shrink-0" />
                            <span className="italic">{entry.responding_to}</span>
                          </p>
                        )}
                        <p className="text-sm text-white/70 leading-relaxed whitespace-pre-wrap pr-14">
                          {entry.text}
                        </p>
                        <div className="absolute top-1.5 right-1.5 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() => focusSuggestion(entry.id)}
                            className="rounded-md p-1.5 text-white/30 hover:text-blue-400 hover:bg-blue-400/10 transition-all"
                            title="Pin this suggestion"
                          >
                            <Pin className="h-3 w-3" />
                          </button>
                          <button
                            onClick={() => handleCopy(entry)}
                            className="rounded-md p-1.5 text-white/30 hover:text-white/80 hover:bg-white/10 transition-all"
                            title="Copy to clipboard"
                          >
                            {copiedId === entry.id ? (
                              <Check className="h-3 w-3 text-emerald-400" />
                            ) : (
                              <Copy className="h-3 w-3" />
                            )}
                          </button>
                        </div>
                        <div className="mt-1 text-[10px] text-white/20">
                          {new Date(entry.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
