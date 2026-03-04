import { useSessionStore } from "@/stores/sessionStore";
import type { SuggestionEntry } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { MessageSquare, Copy, Check, Loader2, ArrowDown, CornerDownLeft, Trash2, X } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";

export function AISuggestionsPanel() {
  const suggestions = useSessionStore((s) => s.suggestions);
  const streaming = useSessionStore((s) => s.aiStreaming);
  const clearSuggestions = useSessionStore((s) => s.clearSuggestions);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [hasNew, setHasNew] = useState(false);
  const prevCountRef = useRef(suggestions.length);

  const checkAtBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    setIsAtBottom(atBottom);
    if (atBottom) setHasNew(false);
  }, []);

  // Auto-scroll when new suggestions arrive (only if already at bottom)
  useEffect(() => {
    if (suggestions.length > prevCountRef.current) {
      if (isAtBottom) {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
      } else {
        setHasNew(true);
      }
    }
    prevCountRef.current = suggestions.length;
  }, [suggestions.length, isAtBottom]);

  const scrollToBottom = () => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    setHasNew(false);
  };

  const handleCopy = async (entry: SuggestionEntry) => {
    try {
      await navigator.clipboard.writeText(entry.text);
      setCopiedId(entry.id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // Clipboard access can fail if window is not focused
    }
  };

  return (
    <div className="flex h-full flex-col">
      <h2 className="flex items-center gap-2 border-b border-white/10 px-4 py-3 text-sm font-semibold text-white/80">
        <MessageSquare className="h-4 w-4" />
        Suggestions
        {streaming && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />}
        {suggestions.length > 0 && (
          <>
            <span className="ml-auto text-xs text-white/30 font-normal">{suggestions.length}</span>
            {confirmClear ? (
              <span className="flex items-center gap-1 text-xs text-white/50 font-normal">
                Clear?
                <button
                  onClick={() => { clearSuggestions(); setConfirmClear(false); }}
                  className="rounded p-0.5 text-red-400 hover:bg-red-400/10 transition-colors"
                  aria-label="Confirm clear suggestions"
                >
                  <Check className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => setConfirmClear(false)}
                  className="rounded p-0.5 text-white/40 hover:text-white/70 transition-colors"
                  aria-label="Cancel clear"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </span>
            ) : (
              <button
                onClick={() => setConfirmClear(true)}
                className="rounded-md p-1 text-white/30 hover:text-white/70 hover:bg-white/5 transition-colors"
                title="Clear all suggestions"
                aria-label="Clear all suggestions"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </>
        )}
      </h2>

      <div className="relative flex-1 min-h-0">
        <div
          ref={scrollRef}
          onScroll={checkAtBottom}
          className="h-full overflow-y-auto p-4 space-y-3"
        >
          {suggestions.length === 0 ? (
            <p className="text-sm text-white/30 text-center py-8">
              Response suggestions will appear here during the conversation.
            </p>
          ) : (
            suggestions.map((entry, idx) => {
              const isLatest = idx === suggestions.length - 1;
              return (
                <div
                  key={entry.id}
                  className={cn(
                    "group relative rounded-lg px-3 py-2.5 transition-colors",
                    isLatest
                      ? "bg-white/[0.05] border border-blue-400/30 border-l-2 border-l-blue-400"
                      : "bg-white/[0.02] border border-white/[0.04] opacity-60",
                  )}
                >
                  {entry.responding_to && (
                    <p className="flex items-start gap-1.5 text-xs text-white/35 mb-1.5 leading-snug">
                      <CornerDownLeft className="h-3 w-3 mt-0.5 shrink-0" />
                      <span className="italic">{entry.responding_to}</span>
                    </p>
                  )}
                  <p className="text-sm text-white/90 leading-relaxed whitespace-pre-wrap pr-8">
                    {entry.text}
                  </p>
                  <button
                    onClick={() => handleCopy(entry)}
                    className={cn(
                      "absolute top-2 right-2 rounded-md p-1.5 text-white/30",
                      "opacity-0 group-hover:opacity-100 hover:text-white/80",
                      "hover:bg-white/10 transition-all",
                    )}
                    title="Copy to clipboard"
                  >
                    {copiedId === entry.id ? (
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </button>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px] text-white/20">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                    {isLatest && (
                      <span className="text-[10px] font-medium text-blue-400/70">Latest</span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {hasNew && (
          <button
            onClick={scrollToBottom}
            className={cn(
              "absolute bottom-3 left-1/2 -translate-x-1/2",
              "flex items-center gap-1 rounded-full bg-blue-500/90 px-3 py-1",
              "text-xs text-white font-medium shadow-lg",
              "hover:bg-blue-500 transition-colors",
            )}
          >
            <ArrowDown className="h-3 w-3" />
            New suggestion
          </button>
        )}
      </div>
    </div>
  );
}
