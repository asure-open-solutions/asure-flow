import { useEffect, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { cn } from "@/lib/utils";
import { ChevronDown, CheckSquare, X, LayoutGrid, Circle, Square, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { Logo } from "../Logo";

export function OverlayTopBar() {
  const transcript = useSessionStore((s) => s.transcript);
  const suggestion = useSessionStore((s) => s.suggestions[s.suggestions.length - 1]?.text ?? null);
  const notes = useSessionStore((s) => s.notes);
  const recording = useSessionStore((s) => s.recording);
  const audioToggles = useSessionStore((s) => s.overlayAudioToggles);
  const overlaySettings = useSettingsStore((s) => s.overlaySettings);
  const setOverlaySettings = useSettingsStore((s) => s.setOverlaySettings);
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const recentTranscript = transcript.slice(-6);
  const actionItems = notes.filter((n) => n.type === "action_item" && !n.completed);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [transcript]);

  const handleMouseEnter = () => {
    window.electronAPI?.setIgnoreMouseEvents(false, false);
  };

  const handleMouseLeave = () => {
    window.electronAPI?.setIgnoreMouseEvents(true, true);
    setExpanded(false);
  };

  const handleClose = () => {
    window.electronAPI?.toggleOverlay();
  };

  const switchToCards = () => {
    setOverlaySettings({ overlayMode: "cards" });
  };

  return (
    <div
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className="absolute top-0 left-0 right-0 flex flex-col items-center pt-2 px-4"
    >
      {/* ── Primary bar: suggestion text (always visible, near camera) ── */}
      <div
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "flex items-start gap-3 rounded-2xl px-4 py-2.5 w-full max-w-2xl",
          "bg-zinc-900/90 backdrop-blur-xl border border-white/10 shadow-2xl",
          "cursor-pointer select-none transition-all",
          "hover:bg-zinc-800/90",
        )}
      >
        <Logo size={14} className="shrink-0 opacity-50 mt-0.5" />

        <div className="flex-1 min-w-0">
          {suggestion ? (
            <p className="text-sm leading-relaxed text-blue-300/90 whitespace-pre-wrap">
              {suggestion}
            </p>
          ) : recentTranscript.length > 0 ? (
            <p className="text-xs text-white/50 truncate">
              {recentTranscript[recentTranscript.length - 1].speaker}:{" "}
              {recentTranscript[recentTranscript.length - 1].text}
            </p>
          ) : (
            <p className="text-xs text-white/25">Waiting for audio...</p>
          )}
        </div>

        <div className="flex items-center gap-1 shrink-0">
          {actionItems.length > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-amber-400/80 bg-amber-400/10 rounded-full px-2 py-0.5">
              <CheckSquare className="h-3 w-3" />
              {actionItems.length}
            </span>
          )}

          {/* Audio toggles */}
          <button
            onClick={(e) => { e.stopPropagation(); window.electronAPI?.setAudioToggle({ mic: !audioToggles.mic }); }}
            className={cn(
              "rounded-md p-1 transition-colors",
              audioToggles.mic
                ? "text-white/25 hover:text-white/60 hover:bg-white/10"
                : "text-red-400/60 hover:text-red-400 hover:bg-red-500/10",
            )}
            title={audioToggles.mic ? "Mute mic" : "Unmute mic"}
          >
            {audioToggles.mic ? <Mic className="h-3 w-3" /> : <MicOff className="h-3 w-3" />}
          </button>

          <button
            onClick={(e) => { e.stopPropagation(); window.electronAPI?.setAudioToggle({ system: !audioToggles.system }); }}
            className={cn(
              "rounded-md p-1 transition-colors",
              audioToggles.system
                ? "text-white/25 hover:text-white/60 hover:bg-white/10"
                : "text-red-400/60 hover:text-red-400 hover:bg-red-500/10",
            )}
            title={audioToggles.system ? "Mute system audio" : "Unmute system audio"}
          >
            {audioToggles.system ? <Volume2 className="h-3 w-3" /> : <VolumeX className="h-3 w-3" />}
          </button>

          {/* Record button */}
          <button
            onClick={(e) => { e.stopPropagation(); window.electronAPI?.toggleRecording(); }}
            className={cn(
              "rounded-md p-1 transition-colors",
              recording
                ? "text-red-400 hover:bg-red-500/20"
                : "text-white/25 hover:text-white/60 hover:bg-white/10",
            )}
            title={recording ? "Stop recording" : "Start recording"}
          >
            {recording ? (
              <Square className="h-3 w-3 fill-red-400" />
            ) : (
              <Circle className="h-3 w-3" />
            )}
          </button>

          <ChevronDown className={cn(
            "h-3.5 w-3.5 text-white/25 transition-transform",
            expanded && "rotate-180",
          )} />

          <button
            onClick={(e) => { e.stopPropagation(); switchToCards(); }}
            className="rounded-md p-1 text-white/25 hover:text-white/60 hover:bg-white/10 transition-colors"
            title="Switch to floating cards"
          >
            <LayoutGrid className="h-3.5 w-3.5" />
          </button>

          <button
            onClick={(e) => { e.stopPropagation(); handleClose(); }}
            className="rounded-md p-1 text-white/25 hover:text-white/60 hover:bg-white/10 transition-colors"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* ── Expanded panel: transcript + notes (click to reveal) ── */}
      {expanded && (
        <div className={cn(
          "mt-1.5 rounded-2xl border border-white/10 bg-zinc-900/95 backdrop-blur-xl shadow-2xl",
          "overflow-hidden w-full max-w-2xl",
        )}>
          {/* Transcript */}
          {overlaySettings.showTranscript && (
            <div ref={scrollRef} className="max-h-44 overflow-y-auto px-4 py-3 space-y-1.5">
              {recentTranscript.length === 0 ? (
                <p className="text-xs text-white/25 text-center py-2">No transcript yet</p>
              ) : (
                recentTranscript.map((entry) => (
                  <div key={entry.id} className="text-xs leading-relaxed">
                    <span className={cn(
                      "font-semibold",
                      entry.speaker === "User" ? "text-blue-400" : "text-violet-400",
                    )}>
                      {entry.speaker}:
                    </span>{" "}
                    <span className="text-white/75">{entry.text}</span>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Action items */}
          {overlaySettings.showNotes && actionItems.length > 0 && (
            <div className="px-4 py-2.5 border-t border-white/[0.06]">
              <div className="flex items-center gap-1 mb-1">
                <CheckSquare className="h-3 w-3 text-amber-400" />
                <span className="text-[10px] font-semibold text-amber-400 uppercase">Action Items</span>
              </div>
              {actionItems.slice(0, 4).map((note) => (
                <p key={note.id} className="text-xs text-white/60 truncate">
                  &bull; {note.content}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
