import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { GripHorizontal, X } from "lucide-react";

interface Props {
  title: string;
  icon: React.ReactNode;
  accentColor?: string;
  initialX?: number;
  initialY?: number;
  onPositionChange?: (x: number, y: number) => void;
  onDismiss?: () => void;
  headerAction?: React.ReactNode;
  children: React.ReactNode;
}

export function OverlayCardWidget({
  title,
  icon,
  accentColor = "text-white/50",
  initialX = 100,
  initialY = 100,
  onPositionChange,
  onDismiss,
  headerAction,
  children,
}: Props) {
  const [pos, setPos] = useState({ x: initialX, y: initialY });
  const dragging = useRef(false);
  const offset = useRef({ x: 0, y: 0 });
  const posRef = useRef(pos);
  const cardRef = useRef<HTMLDivElement>(null);

  // Keep ref in sync so event handlers always read the latest position
  posRef.current = pos;

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    offset.current = { x: e.clientX - posRef.current.x, y: e.clientY - posRef.current.y };
    e.preventDefault();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const newX = e.clientX - offset.current.x;
      const newY = e.clientY - offset.current.y;
      setPos({ x: newX, y: newY });
    };

    const handleMouseUp = () => {
      if (dragging.current) {
        dragging.current = false;
        onPositionChange?.(posRef.current.x, posRef.current.y);
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [onPositionChange]);

  // Sync initial position
  useEffect(() => {
    setPos({ x: initialX, y: initialY });
  }, [initialX, initialY]);

  const handleMouseEnter = () => {
    window.electronAPI?.setIgnoreMouseEvents(false, false);
  };

  const handleMouseLeave = () => {
    if (!dragging.current) {
      window.electronAPI?.setIgnoreMouseEvents(true, true);
    }
  };

  return (
    <div
      ref={cardRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className="absolute"
      style={{ left: pos.x, top: pos.y, maxWidth: 320, minWidth: 200 }}
    >
      <div className={cn(
        "rounded-xl border border-white/10 bg-zinc-900/85 backdrop-blur-xl shadow-2xl",
        "flex flex-col overflow-hidden",
      )}>
        {/* Drag handle */}
        <div
          onMouseDown={handleMouseDown}
          className="flex items-center justify-between px-3 py-1.5 cursor-grab active:cursor-grabbing select-none"
        >
          <div className={cn("flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider", accentColor)}>
            {icon}
            {title}
          </div>
          <div className="flex items-center gap-1">
            {headerAction}
            <GripHorizontal className="h-3 w-3 text-white/20" />
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="rounded p-0.5 text-white/20 hover:text-white/60 hover:bg-white/10 transition-colors"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="max-h-64 overflow-y-auto px-3 pb-2">
          {children}
        </div>
      </div>
    </div>
  );
}
