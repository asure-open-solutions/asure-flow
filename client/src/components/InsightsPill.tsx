import { useSessionStore } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { Sparkles } from "lucide-react";

export function InsightsPill() {
  const unseenCount = useSessionStore((s) => s.unseenInsightCount);
  const setDrawerOpen = useSessionStore((s) => s.setInsightsDrawerOpen);
  const aiStreaming = useSessionStore((s) => s.aiStreaming);

  return (
    <button
      onClick={() => setDrawerOpen(true)}
      className={cn(
        "fixed right-0 top-1/2 -translate-y-1/2 z-20",
        "flex items-center gap-1.5 rounded-l-xl px-2.5 py-3",
        "bg-zinc-800/90 border border-r-0 border-white/10 backdrop-blur-sm",
        "text-white/50 hover:text-white/80 hover:bg-zinc-700/90 transition-all",
        "shadow-lg",
        unseenCount > 0 && "text-blue-400 border-blue-500/30",
      )}
      title="Open Insights (Ctrl+I)"
    >
      <Sparkles className={cn("h-4 w-4", aiStreaming && "animate-pulse text-blue-400")} />
      {unseenCount > 0 && (
        <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-blue-500 px-1 text-[10px] font-bold text-white">
          {unseenCount > 9 ? "9+" : unseenCount}
        </span>
      )}
    </button>
  );
}
