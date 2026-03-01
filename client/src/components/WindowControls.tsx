import { useEffect, useState } from "react";
import { Minus, Square, X, Copy } from "lucide-react";

export function WindowControls() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    window.electronAPI?.windowIsMaximized().then(setMaximized);
    const cleanup = window.electronAPI?.onMaximizeChange(setMaximized);
    return () => cleanup?.();
  }, []);

  return (
    <div className="flex items-center -mr-4 ml-2">
      <button
        onClick={() => window.electronAPI?.windowMinimize()}
        className="flex items-center justify-center h-8 w-11 text-white/40 hover:text-white/80 hover:bg-white/10 transition-colors"
      >
        <Minus className="h-4 w-4" />
      </button>
      <button
        onClick={() => window.electronAPI?.windowMaximize()}
        className="flex items-center justify-center h-8 w-11 text-white/40 hover:text-white/80 hover:bg-white/10 transition-colors"
      >
        {maximized ? <Copy className="h-3.5 w-3.5 -scale-x-100" /> : <Square className="h-3 w-3" />}
      </button>
      <button
        onClick={() => window.electronAPI?.windowClose()}
        className="flex items-center justify-center h-8 w-11 text-white/40 hover:text-white hover:bg-red-500/80 transition-colors"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
