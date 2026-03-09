import { useEffect, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { PanelHeader } from "./PanelHeader";
import { cn } from "@/lib/utils";
import type { AgentLogEntry } from "@/types";
import {
  Activity,
  Brain,
  Wrench,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

function LogIcon({ type }: { type: AgentLogEntry["type"] }) {
  switch (type) {
    case "thinking":
      return <Brain className="h-3 w-3 text-blue-400" />;
    case "tool_call":
      return <Wrench className="h-3 w-3 text-amber-400" />;
    case "tool_result":
      return <CheckCircle className="h-3 w-3 text-emerald-400" />;
    case "done":
      return <CheckCircle className="h-3 w-3 text-white/40" />;
    case "error":
      return <AlertTriangle className="h-3 w-3 text-red-400" />;
    default:
      return null;
  }
}

function LogEntry({ entry }: { entry: AgentLogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const hasDetail = !!entry.detail;

  return (
    <div className="flex flex-col">
      <div
        className={cn(
          "flex items-center gap-2 py-1 px-2 rounded text-xs",
          hasDetail && "cursor-pointer hover:bg-white/5",
        )}
        onClick={() => hasDetail && setExpanded((v) => !v)}
      >
        <LogIcon type={entry.type} />
        <span
          className={cn(
            "flex-1 truncate",
            entry.type === "error" ? "text-red-400" : "text-white/70",
          )}
        >
          {entry.summary}
        </span>
        {hasDetail &&
          (expanded ? (
            <ChevronDown className="h-3 w-3 text-white/30 shrink-0" />
          ) : (
            <ChevronRight className="h-3 w-3 text-white/30 shrink-0" />
          ))}
      </div>
      {expanded && entry.detail && (
        <div className="ml-7 mr-2 mb-1 rounded bg-white/5 px-3 py-2 text-xs text-white/60 whitespace-pre-wrap leading-relaxed max-h-32 overflow-y-auto">
          {entry.detail}
        </div>
      )}
    </div>
  );
}

export function AgentActivityPanel({ showHeader = true }: { showHeader?: boolean }) {
  const agentLog = useSessionStore((s) => s.agentLog);
  const clearAgentLog = useSessionStore((s) => s.clearAgentLog);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [agentLog.length]);

  return (
    <div className="flex h-full flex-col">
      {showHeader && (
        <PanelHeader
          icon={Activity}
          title="Agent Activity"
          count={agentLog.length}
          onClear={clearAgentLog}
          clearAriaLabel="Clear activity log"
        />
      )}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto py-2 px-1"
      >
        {agentLog.length === 0 ? (
          <p className="text-sm text-white/30 text-center py-8">
            AI activity will appear here during processing.
          </p>
        ) : (
          agentLog.map((entry) => (
            <LogEntry key={entry.id} entry={entry} />
          ))
        )}
      </div>
    </div>
  );
}
