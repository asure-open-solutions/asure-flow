import { useEffect, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { listSessions, createSession, deleteSession, getSession, exportSession, exportSessionMarkdown, renameSession } from "@/services/api";
import { downloadFile } from "@/lib/downloadFile";
import { cn } from "@/lib/utils";
import { Plus, Trash2, Clock, FileText, StickyNote, Download, Tag, Pencil, Home } from "lucide-react";

export function SessionList() {
  const sessions = useSessionStore((s) => s.sessions);
  const setSessions = useSessionStore((s) => s.setSessions);
  const setCurrentSession = useSessionStore((s) => s.setCurrentSession);
  const currentSession = useSessionStore((s) => s.currentSession);
  const serverOnline = useSessionStore((s) => s.serverOnline);
  const renameCurrentSessionStore = useSessionStore((s) => s.renameCurrentSession);

  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    try {
      const list = await listSessions();
      setSessions(list);
    } catch (err) {
      console.error("Failed to list sessions:", err);
    }
  };

  // Fetch sessions on mount and whenever the server comes online
  useEffect(() => {
    if (serverOnline) refresh();
  }, [serverOnline]);

  // Focus rename input when entering rename mode
  useEffect(() => {
    if (renamingId) {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }
  }, [renamingId]);

  const handleCreate = async () => {
    try {
      const session = await createSession();
      setCurrentSession(session);
      await refresh();
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  };

  const handleLoad = async (id: string) => {
    if (renamingId) return;
    if (currentSession?.id === id) {
      setCurrentSession(null);
      return;
    }
    try {
      const session = await getSession(id);
      setCurrentSession(session);
    } catch (err) {
      console.error("Failed to load session:", err);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSession(id);
      if (currentSession?.id === id) {
        setCurrentSession(null);
      }
      await refresh();
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const startRename = (id: string, currentName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(id);
    setRenameValue(currentName);
  };

  const handleRename = async (id: string) => {
    const trimmed = renameValue.trim();
    if (!trimmed) {
      setRenamingId(null);
      return;
    }
    try {
      await renameSession(id, trimmed);
      if (currentSession?.id === id) {
        renameCurrentSessionStore(trimmed);
      }
      // Update in session list
      setSessions(sessions.map((s) => (s.id === id ? { ...s, name: trimmed } : s)));
      setRenamingId(null);
    } catch (err) {
      console.error("Failed to rename session:", err);
      // Leave rename mode active so the user can retry
    }
  };

  const [exportingId, setExportingId] = useState<string | null>(null);

  const handleExportJson = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExportingId(id);
    try {
      const session = await exportSession(id);
      downloadFile(JSON.stringify(session, null, 2), `${session.name || "session"}.json`, "application/json");
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExportingId(null);
    }
  };

  const handleExportMarkdown = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExportingId(id);
    try {
      const md = await exportSessionMarkdown(id);
      downloadFile(md, `session-${id.slice(0, 8)}.md`, "text/markdown");
    } catch (err) {
      console.error("Markdown export failed:", err);
    } finally {
      setExportingId(null);
    }
  };

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <h2 className="text-sm font-semibold text-white/80">Sessions</h2>
        <button
          onClick={handleCreate}
          className="flex items-center gap-1 rounded-md bg-blue-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-blue-500 transition-colors"
        >
          <Plus className="h-3.5 w-3.5" />
          New
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        <button
          onClick={() => setCurrentSession(null)}
          className={cn(
            "w-full rounded-lg px-3 py-2 text-left transition-colors flex items-center gap-2",
            "hover:bg-white/5",
            !currentSession
              ? "bg-white/10 ring-1 ring-white/20"
              : "text-white/50",
          )}
        >
          <Home className="h-4 w-4 shrink-0" />
          <span className="text-sm font-medium">Home</span>
        </button>

        {sessions.length === 0 && (
          <p className="text-sm text-white/40 text-center py-8">
            {serverOnline
              ? "No sessions yet. Create one to get started."
              : "Waiting for server..."}
          </p>
        )}

        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => handleLoad(s.id)}
            className={cn(
              "group w-full rounded-lg px-3 py-2.5 text-left transition-colors",
              "hover:bg-white/5",
              currentSession?.id === s.id && "bg-white/10 ring-1 ring-white/20",
            )}
          >
            <div className="flex items-center justify-between">
              {renamingId === s.id ? (
                <input
                  ref={renameInputRef}
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleRename(s.id);
                    if (e.key === "Escape") setRenamingId(null);
                  }}
                  onBlur={() => handleRename(s.id)}
                  onClick={(e) => e.stopPropagation()}
                  className="text-sm font-medium text-white bg-transparent border-b border-white/30 outline-none flex-1 min-w-0 mr-2"
                />
              ) : (
                <span
                  className="text-sm font-medium text-white/90 truncate"
                  onDoubleClick={(e) => startRename(s.id, s.name, e)}
                >
                  {s.name}
                </span>
              )}
              <div className="flex items-center gap-0.5 shrink-0">
                <button
                  onClick={(e) => startRename(s.id, s.name, e)}
                  className="rounded p-1 text-white/30 hover:text-white/70 hover:bg-white/10 transition-colors opacity-0 group-hover:opacity-100"
                  title="Rename"
                >
                  <Pencil className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={(e) => handleExportMarkdown(s.id, e)}
                  disabled={exportingId === s.id}
                  className="rounded p-1 text-white/30 hover:text-blue-400 hover:bg-blue-400/10 transition-colors opacity-0 group-hover:opacity-100"
                  title="Export as Markdown"
                >
                  <Download className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={(e) => handleDelete(s.id, e)}
                  className="rounded p-1 text-white/30 hover:text-red-400 hover:bg-red-400/10 transition-colors opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
            <div className="mt-1 flex items-center gap-3 text-xs text-white/40">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatDate(s.updated_at)}
              </span>
              <span className="flex items-center gap-1">
                <FileText className="h-3 w-3" />
                {s.transcript_count}
              </span>
              <span className="flex items-center gap-1">
                <StickyNote className="h-3 w-3" />
                {s.notes_count}
              </span>
            </div>
            {s.topics && s.topics.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {s.topics.slice(0, 4).map((topic) => (
                  <span
                    key={topic}
                    className="inline-flex items-center gap-0.5 rounded-full bg-white/5 px-1.5 py-0.5 text-[10px] text-white/40"
                  >
                    <Tag className="h-2.5 w-2.5" />
                    {topic}
                  </span>
                ))}
                {s.topics.length > 4 && (
                  <span className="text-[10px] text-white/30">+{s.topics.length - 4}</span>
                )}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
