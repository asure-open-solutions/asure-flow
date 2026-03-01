import { useCallback, useEffect, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { cn } from "@/lib/utils";
import { FileText } from "lucide-react";

interface Props {
  onContextChange: (context: string) => void;
}

export function SessionContextInput({ onContextChange }: Props) {
  const sessionContext = useSessionStore((s) => s.sessionContext);
  const setSessionContext = useSessionStore((s) => s.setSessionContext);

  const [open, setOpen] = useState(false);
  const [localValue, setLocalValue] = useState(sessionContext);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLocalValue(sessionContext);
  }, [sessionContext]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
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

  const hasContext = localValue.trim().length > 0;

  return (
    <div ref={popoverRef} className="relative">
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

      {open && (
        <div className="absolute right-0 top-full z-30 mt-1.5 w-80 rounded-xl border border-white/10 bg-zinc-900 p-3 shadow-2xl">
          <label className="text-xs font-medium text-white/50 mb-1.5 block">Session Context</label>
          <textarea
            value={localValue}
            onChange={(e) => handleChange(e.target.value)}
            placeholder="Describe the situation — who you're talking to, what this is about..."
            rows={3}
            maxLength={1000}
            autoFocus
            className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-xs text-white placeholder-white/20 focus:outline-none focus:ring-1 focus:ring-blue-500/50 resize-none"
          />
          <p className="text-[10px] text-white/25 mt-1">{localValue.length}/1000</p>
        </div>
      )}
    </div>
  );
}
