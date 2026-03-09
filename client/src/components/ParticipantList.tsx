import { useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { PanelHeader } from "./PanelHeader";
import type { Participant } from "@/types";
import { cn } from "@/lib/utils";
import { Users, Pencil, Check, X } from "lucide-react";

function ParticipantRow({
  participant,
  onRename,
}: {
  participant: Participant;
  onRename: (label: string, name: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(participant.display_name);

  const handleSave = () => {
    const trimmed = name.trim();
    if (trimmed && trimmed !== participant.display_name) {
      onRename(participant.speaker_label, trimmed);
    }
    setEditing(false);
  };

  if (editing) {
    return (
      <div className="flex items-center gap-1 rounded-md border border-white/10 bg-white/[0.03] px-2 py-1.5">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSave();
            if (e.key === "Escape") setEditing(false);
          }}
          autoFocus
          className="flex-1 bg-transparent text-sm text-white focus:outline-none"
        />
        <button onClick={handleSave} className="rounded p-0.5 text-emerald-400 hover:bg-emerald-400/10">
          <Check className="h-3 w-3" />
        </button>
        <button onClick={() => setEditing(false)} className="rounded p-0.5 text-white/40 hover:bg-white/10">
          <X className="h-3 w-3" />
        </button>
      </div>
    );
  }

  return (
    <div className="group flex items-center justify-between rounded-md px-2 py-1.5 hover:bg-white/5 transition-colors">
      <div>
        <p className="text-sm text-white/90">{participant.display_name}</p>
        <p className="text-xs text-white/30">
          {participant.speaker_label}
          {participant.role && ` \u00B7 ${participant.role}`}
        </p>
      </div>
      <button
        onClick={() => {
          setName(participant.display_name);
          setEditing(true);
        }}
        className="rounded p-1 text-white/0 group-hover:text-white/40 hover:!text-white/70 transition-colors"
      >
        <Pencil className="h-3 w-3" />
      </button>
    </div>
  );
}

export function ParticipantList({ showHeader = true }: { showHeader?: boolean }) {
  const participants = useSessionStore((s) => s.participants);
  const renameSpeaker = useSessionStore((s) => s.renameSpeaker);
  const sessionContext = useSessionStore((s) => s.sessionContext);

  return (
    <div className="flex h-full flex-col">
      {showHeader && (
        <PanelHeader icon={Users} title="Participants" count={participants.length} />
      )}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {participants.length === 0 ? (
          <p className="text-sm text-white/30 text-center py-8">
            Participants will appear as speakers are detected.
          </p>
        ) : (
          <div className="space-y-0.5">
            {participants.map((p) => (
              <ParticipantRow
                key={p.speaker_label}
                participant={p}
                onRename={renameSpeaker}
              />
            ))}
          </div>
        )}

        {sessionContext && (
          <div className="mt-4 pt-4 border-t border-white/[0.06]">
            <h3 className="text-xs font-semibold text-white/40 mb-2">Session Context</h3>
            <p className="text-xs text-white/60 leading-relaxed whitespace-pre-wrap">{sessionContext}</p>
          </div>
        )}
      </div>
    </div>
  );
}
