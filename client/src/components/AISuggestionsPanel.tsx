import { useSessionStore } from "@/stores/sessionStore";
import type { SuggestionEntry } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { MessageSquare, Copy, Check, Loader2, ArrowDown } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";

export function AISuggestionsPanel() {
  const suggestions = useSessionStore((s) => s.suggestions);
  const streaming = useSessionStore((s) => s.aiStreaming);
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
    await navigator.clipboard.writeText(entry.text);
    setCopiedId(entry.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex h-full flex-col">
      <h2 className="flex items-center gap-2 border-b border-white/10 px-4 py-3 text-sm font-semibold text-white/80">
        <MessageSquare className="h-4 w-4" />
        Suggestions
        {streaming && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />}
        {suggestions.length > 0 && (
          <span className="ml-auto text-xs text-white/30 font-normal">{suggestions.length}</span>
        )}
      </h2>

      <div className="relative flex-1 min-h-0">
        <div
          ref={scrollRef}
          onScroll={checkAtBottom}
          className="h-full overflow-y-auto p-4 space-y-3"
        >
          {suggestions.length === 0 ? (
            <p className="text-sm text-white/40 text-center py-8">
              Response suggestions will appear here during the conversation.
            </p>
          ) : (
            suggestions.map((entry) => (
              <div
                key={entry.id}
                className="group relative rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2.5"
              >
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
                <span className="text-[10px] text-white/20 mt-1 block">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))
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
