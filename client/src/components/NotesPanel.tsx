import { useMemo } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { PanelHeader } from "./PanelHeader";
import type { NoteEntry, NoteType } from "@/types";
import { cn } from "@/lib/utils";
import { CheckSquare, Gavel, Lightbulb, AlertTriangle, StickyNote, User, Calendar, Square, CheckCheck } from "lucide-react";

const NOTE_CONFIG: Record<NoteType, { icon: typeof CheckSquare; label: string; className: string }> = {
  action_item: { icon: CheckSquare, label: "Action Items", className: "text-blue-400 bg-blue-400/10 border-blue-400/20" },
  decision: { icon: Gavel, label: "Decisions", className: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20" },
  key_fact: { icon: Lightbulb, label: "Key Facts", className: "text-amber-400 bg-amber-400/10 border-amber-400/20" },
  risk: { icon: AlertTriangle, label: "Risks", className: "text-red-400 bg-red-400/10 border-red-400/20" },
};

const NOTE_ORDER: NoteType[] = ["action_item", "decision", "key_fact", "risk"];

function NoteItem({
  note,
  config,
}: {
  note: NoteEntry;
  config: { className: string };
}) {
  const isActionItem = note.type === "action_item";
  const toggleNoteCompleted = useSessionStore((s) => s.toggleNoteCompleted);

  return (
    <li
      className={cn(
        "rounded-md border px-3 py-2 text-sm text-white/80",
        config.className,
        note.completed && "opacity-50",
      )}
    >
      <div className="flex items-start gap-2">
        {isActionItem && (
          <button
            onClick={() => toggleNoteCompleted(note.id)}
            className="mt-0.5 shrink-0 text-current"
            title={note.completed ? "Mark incomplete" : "Mark complete"}
          >
            {note.completed ? (
              <CheckCheck className="h-3.5 w-3.5" />
            ) : (
              <Square className="h-3.5 w-3.5" />
            )}
          </button>
        )}
        <span className={cn(note.completed && "line-through")}>{note.content}</span>
      </div>
      {(note.owner || note.due_date) && (
        <div className="mt-1 flex items-center gap-3 text-xs text-white/40">
          {note.owner && (
            <span className="flex items-center gap-1">
              <User className="h-3 w-3" />
              {note.owner}
            </span>
          )}
          {note.due_date && (
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {note.due_date}
            </span>
          )}
        </div>
      )}
    </li>
  );
}

export function NotesPanel({ showHeader = true }: { showHeader?: boolean }) {
  const notes = useSessionStore((s) => s.notes);
  const clearNotes = useSessionStore((s) => s.clearNotes);

  const grouped = useMemo(
    () =>
      NOTE_ORDER.map((type) => ({
        type,
        items: notes.filter((n) => n.type === type),
        config: NOTE_CONFIG[type],
      })).filter((g) => g.items.length > 0),
    [notes],
  );

  return (
    <div className="flex h-full flex-col">
      {showHeader && (
        <PanelHeader
          icon={StickyNote}
          title="Rolling Notes"
          count={notes.length}
          onClear={clearNotes}
          clearAriaLabel="Clear all notes"
        />
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {grouped.length === 0 && (
          <p className="text-sm text-white/30 text-center py-8">
            Notes will appear here as the AI extracts them.
          </p>
        )}

        {grouped.map(({ type, items, config }) => {
          const Icon = config.icon;
          return (
            <div key={type}>
              <h3 className={cn("flex items-center gap-1.5 text-xs font-semibold mb-2", config.className.split(" ")[0])}>
                <Icon className="h-3.5 w-3.5" />
                {config.label}
                <span className="text-white/30 font-normal">({items.length})</span>
              </h3>
              <ul className="space-y-1.5">
                {items.map((note) => (
                  <NoteItem key={note.id} note={note} config={config} />
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}
