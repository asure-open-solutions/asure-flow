import { useSessionStore } from "@/stores/sessionStore";
import type { SuggestionEntry } from "@/stores/sessionStore";
import { PanelHeader } from "./PanelHeader";
import { cn } from "@/lib/utils";
import { MessageSquare, Copy, Check, Loader2, ArrowDown, CornerDownLeft, Crosshair } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";

export function AISuggestionsPanel({ showHeader = true }: { showHeader?: boolean }) {
  const suggestions = useSessionStore((s) => s.suggestions);
  const streaming = useSessionStore((s) => s.aiStreaming);
  const clearSuggestions = useSessionStore((s) => s.clearSuggestions);
  const focusedSuggestionId = useSessionStore((s) => s.focusedSuggestionId);
  const focusSuggestion = useSessionStore((s) => s.focusSuggestion);
  const [copiedId, setCopiedId] = useState<string | null>(null);
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
      {showHeader && (
        <PanelHeader
          icon={MessageSquare}
          title="Suggestions"
          count={suggestions.length}
          onClear={clearSuggestions}
          clearAriaLabel="Clear all suggestions"
          extra={streaming ? <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" /> : undefined}
        />
      )}

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
              const isFocused = focusedSuggestionId === entry.id;
              const isHighlighted = isFocused || isLatest;
              return (
                <div
                  key={entry.id}
                  className={cn(
                    "group relative rounded-lg px-3 py-2.5 transition-colors",
                    isFocused
                      ? "bg-blue-500/10 border border-blue-400/40 border-l-2 border-l-blue-400"
                      : isLatest
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
                  <p className="text-sm text-white/90 leading-relaxed whitespace-pre-wrap pr-16">
                    {entry.text}
                  </p>
                  <div className="absolute top-2 right-2 flex items-center gap-0.5">
                    <button
                      onClick={() => focusSuggestion(entry.id)}
                      className={cn(
                        "rounded-md p-1.5 transition-all",
                        isFocused
                          ? "text-blue-400 opacity-100 bg-blue-400/10"
                          : "text-white/30 opacity-0 group-hover:opacity-100 hover:text-blue-400 hover:bg-blue-400/10",
                      )}
                      title={isFocused ? "Unfocus suggestion" : "Focus suggestion"}
                    >
                      <Crosshair className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => handleCopy(entry)}
                      className={cn(
                        "rounded-md p-1.5 text-white/30",
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
                  </div>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px] text-white/20">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                    {isFocused ? (
                      <span className="text-[10px] font-medium text-blue-400/70">Focused</span>
                    ) : isLatest ? (
                      <span className="text-[10px] font-medium text-blue-400/70">Latest</span>
                    ) : null}
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
