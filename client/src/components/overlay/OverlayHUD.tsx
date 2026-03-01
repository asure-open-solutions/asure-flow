import { useEffect } from "react";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSessionStore } from "@/stores/sessionStore";
import { OverlayTopBar } from "./OverlayTopBar";
import { OverlayCards } from "./OverlayCards";

export function OverlayHUD() {
  const overlayMode = useSettingsStore((s) => s.overlaySettings.overlayMode ?? "topbar");
  const syncFromMain = useSessionStore((s) => s.syncFromMain);

  // Receive session state from the main window via IPC relay
  useEffect(() => {
    const cleanup = window.electronAPI?.onOverlaySync((data: unknown) => {
      const d = data as { transcript: any[]; latestSuggestion: string | null; notes: any[] };
      syncFromMain(d);
    });
    return () => cleanup?.();
  }, [syncFromMain]);

  return (
    <div className="h-screen w-screen" style={{ background: "transparent" }}>
      {overlayMode === "topbar" && <OverlayTopBar />}
      {overlayMode === "cards" && <OverlayCards />}
    </div>
  );
}
