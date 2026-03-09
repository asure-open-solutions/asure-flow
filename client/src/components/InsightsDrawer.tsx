import { useSessionStore } from "@/stores/sessionStore";
import { ClearConfirmButton } from "./ClearConfirmButton";
import { cn } from "@/lib/utils";
import { X, MessageSquare, StickyNote, Users, Activity } from "lucide-react";
import { AISuggestionsPanel } from "./AISuggestionsPanel";
import { NotesPanel } from "./NotesPanel";
import { ParticipantList } from "./ParticipantList";
import { AgentActivityPanel } from "./AgentActivityPanel";
import type { InsightsTab } from "@/types";

const TABS: { id: InsightsTab; label: string; icon: typeof MessageSquare }[] = [
  { id: "suggestions", label: "Suggest", icon: MessageSquare },
  { id: "notes", label: "Notes", icon: StickyNote },
  { id: "people", label: "People", icon: Users },
  { id: "activity", label: "Activity", icon: Activity },
];

/** Map each tab to its store count selector and optional clear action. */
function useTabMeta(tab: InsightsTab) {
  const count = useSessionStore((s) => {
    switch (tab) {
      case "suggestions": return s.suggestions.length;
      case "notes": return s.notes.length;
      case "people": return s.participants.length;
      case "activity": return s.agentLog.length;
    }
  });
  const clearSuggestions = useSessionStore((s) => s.clearSuggestions);
  const clearNotes = useSessionStore((s) => s.clearNotes);
  const clearAgentLog = useSessionStore((s) => s.clearAgentLog);

  const clearFn =
    tab === "suggestions" ? clearSuggestions
    : tab === "notes" ? clearNotes
    : tab === "activity" ? clearAgentLog
    : undefined;

  return { count, clearFn };
}

export function InsightsDrawer() {
  const isOpen = useSessionStore((s) => s.insightsDrawerOpen);
  const activeTab = useSessionStore((s) => s.insightsDrawerTab);
  const setOpen = useSessionStore((s) => s.setInsightsDrawerOpen);
  const setTab = useSessionStore((s) => s.setInsightsDrawerTab);
  const { count: activeCount, clearFn } = useTabMeta(activeTab);

  return (
    <div
      className={cn(
        "shrink-0 flex flex-col border-l border-white/[0.06] bg-zinc-950/50",
        "transition-[width] duration-200 ease-in-out overflow-hidden",
        isOpen ? "w-[360px]" : "w-0",
      )}
    >
      {/* Header with active tab title, count, and clear */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-white/50 uppercase tracking-wider">Insights</span>
          {activeCount > 0 && (
            <span className="text-[10px] text-white/30 font-normal">{activeCount}</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {clearFn && activeCount > 0 && (
            <ClearConfirmButton onClear={clearFn} ariaLabel={`Clear all ${activeTab}`} />
          )}
          <button
            onClick={() => setOpen(false)}
            className="rounded-md p-1 text-white/30 hover:text-white/70 hover:bg-white/5 transition-colors focus-visible:ring-1 focus-visible:ring-white/20 focus-visible:outline-none"
            aria-label="Close insights drawer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/[0.06]">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 py-2 text-xs font-medium transition-colors focus-visible:ring-1 focus-visible:ring-white/20 focus-visible:outline-none",
              activeTab === id
                ? "text-white/90 border-b-2 border-blue-400"
                : "text-white/35 hover:text-white/60",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            <span className="hidden xl:inline">{label}</span>
          </button>
        ))}
      </div>

      {/* Content — headers hidden since tabs + drawer header provide context */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === "suggestions" && <AISuggestionsPanel showHeader={false} />}
        {activeTab === "notes" && <NotesPanel showHeader={false} />}
        {activeTab === "people" && <ParticipantList showHeader={false} />}
        {activeTab === "activity" && <AgentActivityPanel showHeader={false} />}
      </div>
    </div>
  );
}
