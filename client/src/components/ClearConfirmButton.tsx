import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import { Trash2, Check, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface ClearConfirmButtonProps {
  onClear: () => void;
  label?: string;
  icon?: LucideIcon;
  ariaLabel?: string;
  triggerClassName?: string;
}

/** Trash button that requires a second click to confirm. */
export function ClearConfirmButton({
  onClear,
  label = "Clear?",
  icon: Icon = Trash2,
  ariaLabel = "Clear all",
  triggerClassName,
}: ClearConfirmButtonProps) {
  const [confirm, setConfirm] = useState(false);

  if (confirm) {
    return (
      <span className="flex items-center gap-1 text-xs text-white/50 font-normal">
        {label}
        <button
          onClick={() => { onClear(); setConfirm(false); }}
          className="rounded p-0.5 text-red-400 hover:bg-red-400/10 transition-colors"
          aria-label={`Confirm ${ariaLabel.toLowerCase()}`}
        >
          <Check className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => setConfirm(false)}
          className="rounded p-0.5 text-white/40 hover:text-white/70 transition-colors"
          aria-label="Cancel"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </span>
    );
  }

  return (
    <button
      onClick={() => setConfirm(true)}
      className={cn(
        "rounded-md p-1 text-white/30 hover:text-white/70 hover:bg-white/5 transition-colors",
        triggerClassName,
      )}
      title={ariaLabel}
      aria-label={ariaLabel}
    >
      <Icon className="h-3.5 w-3.5" />
    </button>
  );
}
