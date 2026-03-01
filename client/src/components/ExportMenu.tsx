import { useState, useRef, useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { exportSession, exportSessionMarkdown, generateFollowup } from "@/services/api";
import { cn } from "@/lib/utils";
import { Download, FileText, FileJson, Mail, Loader2, Copy, X } from "lucide-react";

export function ExportMenu() {
  const currentSession = useSessionStore((s) => s.currentSession);
  const [open, setOpen] = useState(false);
  const [followupModal, setFollowupModal] = useState<{
    subject: string;
    body: string;
    format: string;
  } | null>(null);
  const [generating, setGenerating] = useState(false);
  const [copied, setCopied] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  if (!currentSession) return null;

  const handleExportJson = async () => {
    setOpen(false);
    try {
      const session = await exportSession(currentSession.id);
      const blob = new Blob([JSON.stringify(session, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${currentSession.name || "session"}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    }
  };

  const handleExportMarkdown = async () => {
    setOpen(false);
    try {
      const md = await exportSessionMarkdown(currentSession.id);
      const blob = new Blob([md], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${currentSession.name || "session"}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Markdown export failed:", err);
    }
  };

  const handleFollowup = async () => {
    setOpen(false);
    setGenerating(true);
    try {
      const result = await generateFollowup(currentSession.id, "email");
      setFollowupModal(result);
    } catch (err) {
      console.error("Follow-up generation failed:", err);
    } finally {
      setGenerating(false);
    }
  };

  const handleCopy = async () => {
    if (!followupModal) return;
    const text = followupModal.subject
      ? `Subject: ${followupModal.subject}\n\n${followupModal.body}`
      : followupModal.body;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <div ref={ref} className="relative">
        <button
          onClick={() => setOpen(!open)}
          disabled={generating}
          className={cn(
            "rounded-md p-1.5 text-white/50 hover:text-white hover:bg-white/10 transition-colors",
            generating && "opacity-50",
          )}
          title="Export & Follow-up"
        >
          {generating ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Download className="h-4 w-4" />
          )}
        </button>

        {open && (
          <div className="absolute right-0 top-full z-30 mt-1 w-48 rounded-lg border border-white/10 bg-zinc-800 py-1 shadow-xl">
            <button
              onClick={handleExportMarkdown}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-white/80 hover:bg-white/5 transition-colors"
            >
              <FileText className="h-4 w-4" />
              Export Markdown
            </button>
            <button
              onClick={handleExportJson}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-white/80 hover:bg-white/5 transition-colors"
            >
              <FileJson className="h-4 w-4" />
              Export JSON
            </button>
            <div className="my-1 border-t border-white/10" />
            <button
              onClick={handleFollowup}
              className="flex w-full items-center gap-2 px-3 py-2 text-sm text-white/80 hover:bg-white/5 transition-colors"
            >
              <Mail className="h-4 w-4" />
              Generate Follow-up
            </button>
          </div>
        )}
      </div>

      {/* Follow-up modal */}
      {followupModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-xl border border-white/10 bg-zinc-900 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
              <h3 className="text-sm font-semibold text-white">Follow-up Draft</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 rounded-md px-2.5 py-1 text-xs text-white/60 hover:text-white hover:bg-white/10 transition-colors"
                >
                  <Copy className="h-3.5 w-3.5" />
                  {copied ? "Copied!" : "Copy"}
                </button>
                <button
                  onClick={() => setFollowupModal(null)}
                  className="rounded-md p-1 text-white/40 hover:text-white hover:bg-white/10 transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
            <div className="p-5 space-y-3">
              {followupModal.subject && (
                <div>
                  <label className="text-xs text-white/40">Subject</label>
                  <p className="text-sm text-white/90 mt-0.5">{followupModal.subject}</p>
                </div>
              )}
              <div>
                <label className="text-xs text-white/40">Body</label>
                <pre className="mt-1 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-md border border-white/10 bg-white/[0.03] p-3 text-sm text-white/80">
                  {followupModal.body}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
