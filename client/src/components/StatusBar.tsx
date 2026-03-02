import { useEffect, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { formatElapsed } from "@/lib/formatElapsed";

function friendlyToolName(name: string): string {
  const map: Record<string, string> = {
    fact_check: "fact-checking",
    suggest_response: "suggesting",
    extract_notes: "extracting notes",
    search_transcript: "searching transcript",
    search_sessions: "searching sessions",
    web_search: "web search",
    format_code: "analysing code",
    deep_think: "deep thinking",
  };
  return map[name] ?? name;
}

export function StatusBar() {
  const serverOnline = useSessionStore((s) => s.serverOnline);
  const llmAvailable = useSessionStore((s) => s.llmAvailable);
  const llmProvider = useSessionStore((s) => s.llmProvider);
  const audioConnected = useSessionStore((s) => s.audioConnected);
  const sessionConnected = useSessionStore((s) => s.sessionConnected);
  const recording = useSessionStore((s) => s.recording);
  const recordingStartedAt = useSessionStore((s) => s.recordingStartedAt);
  const audioWarning = useSessionStore((s) => s.audioWarning);
  const aiStreaming = useSessionStore((s) => s.aiStreaming);
  const currentToolName = useSessionStore((s) => s.currentToolName);

  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!recording || !recordingStartedAt) {
      setElapsed(0);
      return;
    }
    const interval = setInterval(() => {
      setElapsed(Date.now() - recordingStartedAt);
    }, 1000);
    return () => clearInterval(interval);
  }, [recording, recordingStartedAt]);

  let connLabel: string;
  let connColor: string;
  if (!serverOnline) {
    connLabel = "Server offline";
    connColor = "bg-red-400";
  } else if (audioConnected && sessionConnected) {
    connLabel = "Connected";
    connColor = "bg-emerald-400";
  } else if (sessionConnected) {
    connLabel = "No audio";
    connColor = "bg-amber-400";
  } else {
    connLabel = "Server online";
    connColor = "bg-blue-400";
  }

  let aiLabel: string;
  if (currentToolName) {
    aiLabel = `Using ${friendlyToolName(currentToolName)}...`;
  } else if (aiStreaming) {
    aiLabel = "Thinking...";
  } else {
    aiLabel = "Idle";
  }

  return (
    <div className="flex items-center gap-4 border-t border-white/[0.06] bg-zinc-950/80 px-4 py-1 text-[11px] text-white/40 select-none">
      <span className="flex items-center gap-1.5">
        <span className={cn("h-1.5 w-1.5 rounded-full", connColor)} />
        {connLabel}
      </span>

      {serverOnline && (
        <span
          className="flex items-center gap-1.5"
          title={llmAvailable ? `Active provider: ${llmProvider}` : "No LLM provider configured"}
        >
          <span className={cn("h-1.5 w-1.5 rounded-full", llmAvailable ? "bg-emerald-400" : "bg-amber-400")} />
          {llmAvailable ? "LLM" : "No LLM"}
        </span>
      )}

      {recording && (
        <span className="flex items-center gap-1.5 text-red-400">
          <span className="h-1.5 w-1.5 rounded-full bg-red-400 animate-pulse" />
          Recording {formatElapsed(elapsed)}
        </span>
      )}

      {audioWarning && (
        <span className="text-amber-400 truncate max-w-[300px]" title={audioWarning}>
          {audioWarning}
        </span>
      )}

      <span className={cn("ml-auto", aiStreaming && "text-blue-400")}>
        AI: {aiLabel}
      </span>
    </div>
  );
}
