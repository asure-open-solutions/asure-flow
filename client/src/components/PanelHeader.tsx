import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { ClearConfirmButton } from "./ClearConfirmButton";

interface PanelHeaderProps {
  icon: LucideIcon;
  title: string;
  count?: number;
  onClear?: () => void;
  clearIcon?: LucideIcon;
  clearAriaLabel?: string;
  extra?: ReactNode;
}

/** Standardized header bar for insight panels. */
export function PanelHeader({
  icon: Icon,
  title,
  count,
  onClear,
  clearIcon,
  clearAriaLabel,
  extra,
}: PanelHeaderProps) {
  return (
    <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3 text-sm font-semibold text-white/80">
      <Icon className="h-4 w-4" />
      {title}
      {extra}
      {count != null && count > 0 && (
        <>
          <span className="ml-auto text-xs text-white/30 font-normal">{count}</span>
          {onClear && (
            <ClearConfirmButton
              onClear={onClear}
              icon={clearIcon}
              ariaLabel={clearAriaLabel ?? `Clear all ${title.toLowerCase()}`}
            />
          )}
        </>
      )}
    </div>
  );
}
