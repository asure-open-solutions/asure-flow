import { useSessionStore } from "@/stores/sessionStore";

interface ConnectionStatus {
  label: string;
  dotColor: string;
  isOffline: boolean;
}

/** Derive connection status from store — replaces duplicate logic in StatusBar & HeaderMoreMenu. */
export function useConnectionStatus(): ConnectionStatus {
  const serverOnline = useSessionStore((s) => s.serverOnline);
  const audioConnected = useSessionStore((s) => s.audioConnected);
  const sessionConnected = useSessionStore((s) => s.sessionConnected);

  if (!serverOnline) {
    return { label: "Server offline", dotColor: "bg-red-400", isOffline: true };
  }
  if (audioConnected && sessionConnected) {
    return { label: "Connected", dotColor: "bg-emerald-400", isOffline: false };
  }
  if (sessionConnected) {
    return { label: "No audio", dotColor: "bg-amber-400", isOffline: false };
  }
  return { label: "Server online", dotColor: "bg-blue-400", isOffline: false };
}
