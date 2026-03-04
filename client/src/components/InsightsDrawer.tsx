import { useSessionStore } from "@/stores/sessionStore";
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

export function InsightsDrawer() {
  const isOpen = useSessionStore((s) => s.insightsDrawerOpen);
  const activeTab = useSessionStore((s) => s.insightsDrawerTab);
  const setOpen = useSessionStore((s) => s.setInsightsDrawerOpen);
  const setTab = useSessionStore((s) => s.setInsightsDrawerTab);

  return (
    <div
      className={cn(
        "shrink-0 flex flex-col border-l border-white/[0.06] bg-zinc-950/50",
        "transition-[width] duration-200 ease-in-out overflow-hidden",
        isOpen ? "w-[360px]" : "w-0",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
        <span className="text-xs font-semibold text-white/50 uppercase tracking-wider">Insights</span>
        <button
          onClick={() => setOpen(false)}
          className="rounded-md p-1 text-white/30 hover:text-white/70 hover:bg-white/5 transition-colors focus-visible:ring-1 focus-visible:ring-white/20 focus-visible:outline-none"
          aria-label="Close insights drawer"
        >
          <X className="h-3.5 w-3.5" />
        </button>
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

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === "suggestions" && <AISuggestionsPanel />}
        {activeTab === "notes" && <NotesPanel />}
        {activeTab === "people" && <ParticipantList />}
        {activeTab === "activity" && <AgentActivityPanel />}
      </div>
    </div>
  );
}
