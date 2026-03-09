import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { OverlayCardWidget } from "./OverlayCardWidget";
import { FactCheckBadge } from "../FactCheckBadge";
import { cn } from "@/lib/utils";
import { MessageSquare, CheckSquare, AlignLeft, X, Monitor, Circle, Square, Mic, MicOff, Volume2, VolumeX, Crosshair } from "lucide-react";
import { Logo } from "../Logo";

export function OverlayCards() {
  const transcript = useSessionStore((s) => s.transcript);
  const focusedSuggestionId = useSessionStore((s) => s.focusedSuggestionId);
  const focusedEntry = useSessionStore((s) =>
    s.focusedSuggestionId ? s.suggestions.find((x) => x.id === s.focusedSuggestionId) : null,
  );
  const latestEntry = useSessionStore((s) => s.suggestions[s.suggestions.length - 1] ?? null);
  const displayEntry = focusedEntry ?? latestEntry;
  const suggestion = displayEntry?.text ?? null;
  const isFocused = !!focusedEntry;
  const notes = useSessionStore((s) => s.notes);
  const recording = useSessionStore((s) => s.recording);
  const audioToggles = useSessionStore((s) => s.overlayAudioToggles);
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
      {/* Controls — top-right corner */}
      <div
        className="absolute top-2 right-4 flex items-center gap-1 z-50"
        onMouseEnter={() => window.electronAPI?.setIgnoreMouseEvents(false, false)}
        onMouseLeave={() => window.electronAPI?.setIgnoreMouseEvents(true, true)}
      >
        <div className="flex items-center gap-1 rounded-xl bg-zinc-900/85 backdrop-blur-xl border border-white/10 px-2 py-1 shadow-lg">
          <Logo size={14} className="opacity-50" />

          {/* Audio toggles */}
          <button
            onClick={() => window.electronAPI?.setAudioToggle({ mic: !audioToggles.mic })}
            className={cn(
              "rounded-md p-1 transition-colors",
              audioToggles.mic
                ? "text-white/30 hover:text-white/70 hover:bg-white/10"
                : "text-red-400/60 hover:text-red-400 hover:bg-red-500/10",
            )}
            title={audioToggles.mic ? "Mute mic" : "Unmute mic"}
          >
            {audioToggles.mic ? <Mic className="h-3.5 w-3.5" /> : <MicOff className="h-3.5 w-3.5" />}
          </button>

          <button
            onClick={() => window.electronAPI?.setAudioToggle({ system: !audioToggles.system })}
            className={cn(
              "rounded-md p-1 transition-colors",
              audioToggles.system
                ? "text-white/30 hover:text-white/70 hover:bg-white/10"
                : "text-red-400/60 hover:text-red-400 hover:bg-red-500/10",
            )}
            title={audioToggles.system ? "Mute system audio" : "Unmute system audio"}
          >
            {audioToggles.system ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
          </button>

          {/* Record button */}
          <button
            onClick={() => window.electronAPI?.toggleRecording()}
            className={cn(
              "rounded-md p-1 transition-colors",
              recording
                ? "text-red-400 hover:bg-red-500/20"
                : "text-white/30 hover:text-white/70 hover:bg-white/10",
            )}
            title={recording ? "Stop recording" : "Start recording"}
          >
            {recording ? (
              <Square className="h-3.5 w-3.5 fill-red-400" />
            ) : (
              <Circle className="h-3.5 w-3.5" />
            )}
          </button>

          <div className="w-px h-3 bg-white/10 mx-0.5" />

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
          title={isFocused ? "Suggestion (Focused)" : "Suggestion"}
          icon={<MessageSquare className="h-3 w-3" />}
          accentColor="text-blue-400"
          initialX={getPos("suggestion").x}
          initialY={getPos("suggestion").y}
          onPositionChange={(x, y) => updateCardPosition("suggestion", x, y)}
          headerAction={
            displayEntry ? (
              <button
                onClick={() => {
                  const id = isFocused ? null : displayEntry.id;
                  window.electronAPI?.focusSuggestion(id);
                }}
                className={cn(
                  "rounded p-0.5 transition-colors",
                  isFocused
                    ? "text-blue-400 hover:bg-blue-400/20"
                    : "text-white/30 hover:text-blue-400 hover:bg-blue-400/10",
                )}
                title={isFocused ? "Unfocus" : "Focus suggestion"}
              >
                <Crosshair className="h-3 w-3" />
              </button>
            ) : undefined
          }
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
