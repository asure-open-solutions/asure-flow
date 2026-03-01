import { useSessionStore } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { MessageSquare, Copy, Check, Loader2 } from "lucide-react";
import { useState } from "react";

export function AISuggestionsPanel() {
  const suggestion = useSessionStore((s) => s.latestSuggestion);
  const streaming = useSessionStore((s) => s.aiStreaming);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!suggestion) return;
    await navigator.clipboard.writeText(suggestion);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex h-full flex-col">
      <h2 className="flex items-center gap-2 border-b border-white/10 px-4 py-3 text-sm font-semibold text-white/80">
        <MessageSquare className="h-4 w-4" />
        AI Suggestion
        {streaming && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />}
      </h2>

      <div className="flex-1 overflow-y-auto p-4">
        {suggestion ? (
          <div className="group relative">
            <p className="text-sm text-white/90 leading-relaxed whitespace-pre-wrap">
              {suggestion}
            </p>
            <button
              onClick={handleCopy}
              className={cn(
                "absolute top-0 right-0 rounded-md p-1.5 text-white/40 hover:text-white/80",
                "hover:bg-white/10 transition-colors",
              )}
              title="Copy to clipboard"
            >
              {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
            </button>
          </div>
        ) : (
          <p className="text-sm text-white/40 text-center py-8">
            Response suggestions will appear here during the conversation.
          </p>
        )}
      </div>
    </div>
  );
}
