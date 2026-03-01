import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { OverlayCardWidget } from "./OverlayCardWidget";
import { FactCheckBadge } from "../FactCheckBadge";
import { cn } from "@/lib/utils";
import { MessageSquare, CheckSquare, AlignLeft, X, Monitor } from "lucide-react";
import { Logo } from "../Logo";

export function OverlayCards() {
  const transcript = useSessionStore((s) => s.transcript);
  const suggestion = useSessionStore((s) => s.latestSuggestion);
  const notes = useSessionStore((s) => s.notes);
  const overlaySettings = useSettingsStore((s) => s.overlaySettings);
  const setOverlaySettings = useSettingsStore((s) => s.setOverlaySettings);
  const scrollRef = useRef<HTMLDivElement>(null);

  const recentTranscript = transcript.slice(-6);
  const actionItems = notes.filter((n) => n.type === "action_item" && !n.completed);
  const recentFactChecks = transcript.flatMap((t) => t.fact_checks).slice(-3);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [transcript]);

  const handleClose = () => {
    window.electronAPI?.toggleOverlay();
  };

  const switchToTopBar = () => {
    setOverlaySettings({ overlayMode: "topbar" });
  };

  const updateCardPosition = (key: "transcript" | "suggestion" | "notes", x: number, y: number) => {
    setOverlaySettings({
      cardPositions: {
        ...overlaySettings.cardPositions,
        [key]: { x, y },
      },
    });
  };

  // Default positions based on screen size
  const defaultPositions = {
    transcript: { x: 40, y: 100 },
    suggestion: { x: 400, y: 60 },
    notes: { x: 400, y: 300 },
  };

  const getPos = (key: "transcript" | "suggestion" | "notes") =>
    overlaySettings.cardPositions[key] ?? defaultPositions[key];

  return (
    <div className="h-screen w-screen" style={{ background: "transparent" }}>
      {/* Mode switch + close — top-right corner */}
      <div
        className="absolute top-2 right-4 flex items-center gap-1 z-50"
        onMouseEnter={() => window.electronAPI?.setIgnoreMouseEvents(false, false)}
        onMouseLeave={() => window.electronAPI?.setIgnoreMouseEvents(true, true)}
      >
        <div className="flex items-center gap-1 rounded-xl bg-zinc-900/85 backdrop-blur-xl border border-white/10 px-2 py-1 shadow-lg">
          <Logo size={14} className="opacity-50" />
          <button
            onClick={switchToTopBar}
            className="rounded-md p-1 text-white/30 hover:text-white/70 hover:bg-white/10 transition-colors"
            title="Switch to top bar"
          >
            <Monitor className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleClose}
            className="rounded-md p-1 text-white/30 hover:text-white/70 hover:bg-white/10 transition-colors"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Transcript card */}
      {overlaySettings.showTranscript && (
        <OverlayCardWidget
          title="Transcript"
          icon={<AlignLeft className="h-3 w-3" />}
          accentColor="text-violet-400"
          initialX={getPos("transcript").x}
          initialY={getPos("transcript").y}
          onPositionChange={(x, y) => updateCardPosition("transcript", x, y)}
        >
          <div ref={scrollRef} className="space-y-1.5">
            {recentTranscript.length === 0 ? (
              <p className="text-xs text-white/25 text-center py-3">Waiting for audio...</p>
            ) : (
              recentTranscript.map((entry) => (
                <div key={entry.id} className="text-xs leading-relaxed">
                  <span className={cn("font-semibold", entry.speaker === "User" ? "text-blue-400" : "text-violet-400")}>
                    {entry.speaker}:
                  </span>{" "}
                  <span className="text-white/75">{entry.text}</span>
                </div>
              ))
            )}
          </div>

          {overlaySettings.showFactChecks && recentFactChecks.length > 0 && (
            <div className="mt-2 pt-2 border-t border-white/[0.06] flex flex-wrap gap-1">
              {recentFactChecks.map((fc, i) => (
                <FactCheckBadge key={i} check={fc} />
              ))}
            </div>
          )}
        </OverlayCardWidget>
      )}

      {/* Suggestion card */}
      {overlaySettings.showSuggestions && suggestion && (
        <OverlayCardWidget
          title="Suggestion"
          icon={<MessageSquare className="h-3 w-3" />}
          accentColor="text-blue-400"
          initialX={getPos("suggestion").x}
          initialY={getPos("suggestion").y}
          onPositionChange={(x, y) => updateCardPosition("suggestion", x, y)}
        >
          <p className="text-xs text-white/75 leading-relaxed whitespace-pre-wrap">{suggestion}</p>
        </OverlayCardWidget>
      )}

      {/* Action items card */}
      {overlaySettings.showNotes && actionItems.length > 0 && (
        <OverlayCardWidget
          title="Action Items"
          icon={<CheckSquare className="h-3 w-3" />}
          accentColor="text-amber-400"
          initialX={getPos("notes").x}
          initialY={getPos("notes").y}
          onPositionChange={(x, y) => updateCardPosition("notes", x, y)}
        >
          <div className="space-y-1">
            {actionItems.slice(0, 6).map((note) => (
              <p key={note.id} className="text-xs text-white/60">
                &bull; {note.content}
              </p>
            ))}
          </div>
        </OverlayCardWidget>
      )}
    </div>
  );
}
