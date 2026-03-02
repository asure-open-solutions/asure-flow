import { useCallback, useEffect, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { FileText, Upload, X } from "lucide-react";

interface Props {
  onContextChange: (context: string) => void;
}

const ACCEPTED_EXTENSIONS = /\.(txt|md|text)$/i;

export function SessionContextInput({ onContextChange }: Props) {
  const sessionContext = useSessionStore((s) => s.sessionContext);
  const setSessionContext = useSessionStore((s) => s.setSessionContext);

  const [open, setOpen] = useState(false);
  const [localValue, setLocalValue] = useState(sessionContext);
  const [dragging, setDragging] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setLocalValue(sessionContext);
  }, [sessionContext]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open]);

  const handleChange = useCallback(
    (value: string) => {
      setLocalValue(value);
      setSessionContext(value);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onContextChange(value);
      }, 500);
    },
    [onContextChange, setSessionContext],
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const readFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = () => handleChange(reader.result as string);
      reader.onerror = () => console.error("Failed to read file:", file.name);
      reader.readAsText(file);
    },
    [handleChange],
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) readFile(file);
      e.target.value = "";
    },
    [readFile],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && ACCEPTED_EXTENSIONS.test(file.name)) {
        readFile(file);
      }
    },
    [readFile],
  );

  const hasContext = localValue.trim().length > 0;

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "rounded-lg p-1.5 transition-colors",
          hasContext
            ? "text-blue-400/70 hover:text-blue-400"
            : "text-white/40 hover:text-white/80",
          open && "bg-white/5",
          "hover:bg-white/5",
        )}
        title={hasContext ? "Edit session context" : "Add session context"}
      >
        <FileText className="h-4 w-4" />
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.md,.text"
        className="hidden"
        onChange={handleFileSelect}
      />

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setOpen(false)}
          style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
        >
          <div
            className="w-full max-w-xl rounded-xl border border-white/10 bg-zinc-900 shadow-2xl flex flex-col max-h-[70vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-3 shrink-0">
              <h3 className="text-sm font-semibold text-white">Session Context</h3>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs text-white/50 hover:text-white/80 hover:bg-white/5 transition-colors"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Import file
                </button>
                <button
                  onClick={() => setOpen(false)}
                  className="rounded-lg p-1 text-white/40 hover:text-white/80 hover:bg-white/5 transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Body */}
            <div
              className="relative flex-1 min-h-0 p-4"
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              {dragging && (
                <div className="absolute inset-4 z-10 flex items-center justify-center rounded-lg border-2 border-dashed border-blue-400/50 bg-blue-500/10 pointer-events-none">
                  <p className="text-sm text-blue-400/80">Drop file here</p>
                </div>
              )}
              <textarea
                value={localValue}
                onChange={(e) => handleChange(e.target.value)}
                placeholder="Describe the situation — who you're talking to, what this is about...&#10;&#10;You can also paste or import a document for additional context."
                autoFocus
                className="w-full h-full min-h-[200px] rounded-lg bg-white/5 border border-white/10 px-3 py-2.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-1 focus:ring-blue-500/50 resize-y"
              />
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between border-t border-white/10 px-5 py-2.5 shrink-0">
              <p className="text-[11px] text-white/25">
                {localValue.length.toLocaleString()} chars
              </p>
              <p className="text-[11px] text-white/25">
                Drag &amp; drop .txt or .md files
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
