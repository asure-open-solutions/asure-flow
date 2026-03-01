import { cn } from "@/lib/utils";
import type { FactCheck } from "@/types";
import { CheckCircle, XCircle, HelpCircle } from "lucide-react";

const VERDICT_CONFIG = {
  supported: { icon: CheckCircle, className: "text-emerald-400 bg-emerald-400/10", label: "Supported" },
  contradicted: { icon: XCircle, className: "text-red-400 bg-red-400/10", label: "Contradicted" },
  uncertain: { icon: HelpCircle, className: "text-amber-400 bg-amber-400/10", label: "Uncertain" },
} as const;

export function FactCheckBadge({ check }: { check: FactCheck }) {
  const config = VERDICT_CONFIG[check.verdict] ?? VERDICT_CONFIG.uncertain;
  const Icon = config.icon;

  return (
    <div className="inline-flex items-center gap-1.5">
      <div
        className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium", config.className)}
        title={check.reasoning}
      >
        <Icon className="h-3.5 w-3.5" />
        <span className="max-w-48 truncate">{check.claim}</span>
      </div>
      {check.fallacy && (
        <span
          className="inline-flex items-center rounded-full bg-purple-400/10 px-2 py-0.5 text-[10px] font-medium text-purple-400"
          title={`Logical fallacy: ${check.fallacy}`}
        >
          {check.fallacy}
        </span>
      )}
    </div>
  );
}
