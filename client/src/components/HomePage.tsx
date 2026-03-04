import type { SessionSummary } from "@/types";
import { Logo } from "@/components/Logo";
import { Plus, FileText, StickyNote, Clock, Tag, MessageSquare } from "lucide-react";

interface HomePageProps {
  sessions: SessionSummary[];
  onCreateSession: () => void;
  onSelectSession: (id: string) => void;
}

const formatDate = (iso: string) => {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};

export function HomePage({ sessions, onCreateSession, onSelectSession }: HomePageProps) {
  if (sessions.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center max-w-sm">
          <Logo size={40} className="mx-auto mb-4 opacity-20" />
          <h2 className="text-base font-semibold text-white/50 mb-2">Start a Conversation</h2>
          <p className="text-sm text-white/30 mb-6">
            Create a new session to begin capturing and analyzing conversations in real time.
          </p>
          <button
            onClick={onCreateSession}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-500/15 border border-blue-500/20 px-4 py-2 text-sm text-blue-400 hover:bg-blue-500/25 transition-colors"
          >
            <Plus className="h-4 w-4" />
            New Session
          </button>
        </div>
      </div>
    );
  }

  const totalTranscripts = sessions.reduce((sum, s) => sum + s.transcript_count, 0);
  const totalNotes = sessions.reduce((sum, s) => sum + s.notes_count, 0);
  const recentSessions = [...sessions]
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    .slice(0, 5);

  return (
    <div className="flex flex-1 items-start justify-center overflow-y-auto">
      <div className="w-full max-w-2xl px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-base font-semibold text-white/80">Overview</h2>
          <button
            onClick={onCreateSession}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-500/15 border border-blue-500/20 px-3 py-1.5 text-sm text-blue-400 hover:bg-blue-500/25 transition-colors"
          >
            <Plus className="h-3.5 w-3.5" />
            New Session
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-8">
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
            <div className="flex items-center gap-1.5 text-xs text-white/40 mb-1">
              <MessageSquare className="h-3 w-3" />
              Sessions
            </div>
            <p className="text-lg font-semibold text-white/90">{sessions.length}</p>
          </div>
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
            <div className="flex items-center gap-1.5 text-xs text-white/40 mb-1">
              <FileText className="h-3 w-3" />
              Transcripts
            </div>
            <p className="text-lg font-semibold text-white/90">{totalTranscripts}</p>
          </div>
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
            <div className="flex items-center gap-1.5 text-xs text-white/40 mb-1">
              <StickyNote className="h-3 w-3" />
              Notes
            </div>
            <p className="text-lg font-semibold text-white/90">{totalNotes}</p>
          </div>
        </div>

        {/* Recent Sessions */}
        <h3 className="text-sm font-semibold text-white/60 mb-3">Recent Sessions</h3>
        <div className="space-y-1.5">
          {recentSessions.map((s) => (
            <button
              key={s.id}
              onClick={() => onSelectSession(s.id)}
              className="group w-full rounded-lg px-4 py-3 text-left transition-colors hover:bg-white/5 border border-transparent hover:border-white/[0.06]"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-white/90 truncate">{s.name}</span>
                <span className="flex items-center gap-1 text-xs text-white/30 shrink-0 ml-3">
                  <Clock className="h-3 w-3" />
                  {formatDate(s.updated_at)}
                </span>
              </div>
              <div className="mt-1.5 flex items-center gap-4">
                <span className="flex items-center gap-1 text-xs text-white/40">
                  <FileText className="h-3 w-3" />
                  {s.transcript_count}
                </span>
                <span className="flex items-center gap-1 text-xs text-white/40">
                  <StickyNote className="h-3 w-3" />
                  {s.notes_count}
                </span>
                {s.topics && s.topics.length > 0 && (
                  <div className="flex items-center gap-1 flex-wrap">
                    {s.topics.slice(0, 3).map((topic) => (
                      <span
                        key={topic}
                        className="inline-flex items-center gap-0.5 rounded-full bg-white/5 px-1.5 py-0.5 text-[10px] text-white/40"
                      >
                        <Tag className="h-2.5 w-2.5" />
                        {topic}
                      </span>
                    ))}
                    {s.topics.length > 3 && (
                      <span className="text-[10px] text-white/30">+{s.topics.length - 3}</span>
                    )}
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
