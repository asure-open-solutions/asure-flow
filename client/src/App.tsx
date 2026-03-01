import { useCallback, useEffect, useRef, useState } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import { SessionList } from "@/components/SessionList";
import { SettingsView } from "@/components/SettingsView";
import { StatusBar } from "@/components/StatusBar";
import { SessionContextInput } from "@/components/SessionContextInput";
import { HeaderMoreMenu } from "@/components/HeaderMoreMenu";
import { InsightsDrawer } from "@/components/InsightsDrawer";
import { InsightsPill } from "@/components/InsightsPill";
import { OverlayHUD } from "@/components/overlay/OverlayHUD";
import { WindowControls } from "@/components/WindowControls";
import { AudioCapture } from "@/services/audioCapture";
import { AudioWebSocket, SessionWebSocket } from "@/services/websocket";
import { setServerUrl, getServerConfig, isServerReachable, renameSession } from "@/services/api";
import { cn } from "@/lib/utils";
import {
  Mic,
  Settings,
  Square,
  ServerOff,
  Loader2,
  Plus,
} from "lucide-react";
import { Logo } from "@/components/Logo";

function isOverlayRoute() {
  return window.location.hash === "#/overlay";
}

function MainApp() {
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingHeaderName, setEditingHeaderName] = useState(false);
  const [headerNameValue, setHeaderNameValue] = useState("");
  const currentSession = useSessionStore((s) => s.currentSession);
  const renameCurrentSession = useSessionStore((s) => s.renameCurrentSession);
  const recording = useSessionStore((s) => s.recording);
  const serverOnline = useSessionStore((s) => s.serverOnline);
  const setServerOnline = useSessionStore((s) => s.setServerOnline);
  const setRecording = useSessionStore((s) => s.setRecording);
  const setAudioConnected = useSessionStore((s) => s.setAudioConnected);
  const setSessionConnected = useSessionStore((s) => s.setSessionConnected);
  const addTranscriptEntry = useSessionStore((s) => s.addTranscriptEntry);
  const relabelSpeaker = useSessionStore((s) => s.relabelSpeaker);
  const handleAIEvent = useSessionStore((s) => s.handleAIEvent);
  const clearAgentLog = useSessionStore((s) => s.clearAgentLog);
  const insightsDrawerOpen = useSessionStore((s) => s.insightsDrawerOpen);
  const setInsightsDrawerOpen = useSessionStore((s) => s.setInsightsDrawerOpen);

  const serverUrl = useSettingsStore((s) => s.serverUrl);
  const effectiveToggles = useSettingsStore((s) => s.getEffectiveToggles());
  const audioToggles = useSettingsStore((s) => s.audioToggles);
  const contentProtection = useSettingsStore((s) => s.overlaySettings.contentProtection);

  const audioWsRef = useRef<AudioWebSocket | null>(null);
  const sessionWsRef = useRef<SessionWebSocket | null>(null);
  const audioCaptureRef = useRef<AudioCapture | null>(null);
  const serverHandlesSystemRef = useRef(false);

  // Keep server URL in sync
  useEffect(() => {
    setServerUrl(serverUrl);
  }, [serverUrl]);

  // Poll server health
  useEffect(() => {
    let active = true;
    const check = async () => {
      const online = await isServerReachable();
      if (active) setServerOnline(online);
    };
    check();
    const ms = serverOnline ? 30_000 : 3_000;
    const id = setInterval(check, ms);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [serverOnline, serverUrl]);

  // Connect WebSockets when session changes
  useEffect(() => {
    sessionWsRef.current?.disconnect();
    sessionWsRef.current = null;

    if (!currentSession) return;

    const sws = new SessionWebSocket(currentSession.id);
    sws.onAIEvent = (event) => {
      const store = useSessionStore.getState();
      const lastEntry = store.agentLog[store.agentLog.length - 1];
      if (
        event.type === "content_delta" &&
        !store.aiStreaming &&
        lastEntry?.type === "done"
      ) {
        clearAgentLog();
      }
      handleAIEvent(event);
    };
    sws.onSpeakerRenamed = (data) => {
      const store = useSessionStore.getState();
      store.renameSpeaker(data.speaker_label, data.display_name);
      store.updateParticipant(data.participant);
    };
    sws.onConnectionChange = setSessionConnected;
    sws.connect();
    sessionWsRef.current = sws;

    sws.sendConfig(useSettingsStore.getState().getEffectiveToggles());

    return () => {
      sws.disconnect();
    };
  }, [currentSession?.id]);

  // Sync effective feature toggles
  useEffect(() => {
    sessionWsRef.current?.sendConfig(effectiveToggles);
  }, [effectiveToggles]);

  // Sync audio toggles while recording
  useEffect(() => {
    const capture = audioCaptureRef.current;
    if (!capture || !recording) return;

    if (audioToggles.mic) {
      capture.enableMic();
    } else {
      capture.disableMic();
    }

    // Skip client-side system capture when server handles it via loopback
    if (!serverHandlesSystemRef.current) {
      if (audioToggles.system) {
        capture.enableSystem().catch(() => {});
      } else {
        capture.disableSystem();
      }
    }
  }, [audioToggles.mic, audioToggles.system, recording]);

  // Sync content protection
  useEffect(() => {
    window.electronAPI?.setContentProtection(contentProtection);
  }, [contentProtection]);

  // Sync session state to overlay window via IPC
  useEffect(() => {
    const sendSync = () => {
      const { transcript, latestSuggestion, notes } = useSessionStore.getState();
      window.electronAPI?.sendOverlaySync({
        transcript: transcript.slice(-20),
        latestSuggestion,
        notes,
      });
    };

    // Subscribe to store changes — relay to overlay
    const unsub = useSessionStore.subscribe(sendSync);

    // When overlay opens, send current state immediately
    const cleanupOpened = window.electronAPI?.onOverlayOpened(sendSync);

    return () => {
      unsub();
      cleanupOpened?.();
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "b") {
        e.preventDefault();
        setShowSidebar((v) => !v);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "i") {
        e.preventDefault();
        setInsightsDrawerOpen(!insightsDrawerOpen);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [insightsDrawerOpen, setInsightsDrawerOpen]);

  const handleSessionContextChange = useCallback(
    (context: string) => {
      sessionWsRef.current?.sendSessionContext(context);
    },
    [],
  );

  const startRecording = useCallback(async () => {
    if (recording) return;

    let micDeviceId: string | undefined;
    let serverHandlesSystem = false;
    try {
      const cfg = await getServerConfig();
      if (cfg.audio_capture_source === "client" && cfg.mic_device_id) {
        micDeviceId = cfg.mic_device_id;
      }
      // If system_device_id is set, server handles system audio via loopback
      serverHandlesSystem = !!cfg.system_device_id;
    } catch {
      // Proceed with system default
    }
    serverHandlesSystemRef.current = serverHandlesSystem;

    const aws = new AudioWebSocket();
    aws.onConnectionChange = setAudioConnected;
    aws.onTranscription = (data) => {
      const id = crypto.randomUUID().replace(/-/g, "").slice(0, 12);
      addTranscriptEntry({
        id,
        speaker: data.speaker,
        text: data.text,
        timestamp: new Date().toISOString(),
        audio_start: data.start,
        audio_end: data.end,
      });
      sessionWsRef.current?.sendTranscription(id, data.speaker, data.text, data.start, data.end);
    };
    aws.onRelabel = (data) => {
      relabelSpeaker(data.entry_id, data.speaker);
      sessionWsRef.current?.sendRelabel(data.entry_id, data.speaker);
    };
    aws.connect();
    audioWsRef.current = aws;

    const capture = new AudioCapture();
    capture.onChunk = (streamId, pcmData) => {
      aws.sendAudio(streamId, pcmData);
    };
    await capture.start({
      mic: audioToggles.mic,
      system: audioToggles.system && !serverHandlesSystem,
      micDeviceId,
    });
    audioCaptureRef.current = capture;

    setRecording(true);
  }, [recording, audioToggles, addTranscriptEntry, relabelSpeaker, setAudioConnected, setRecording]);

  const stopRecording = useCallback(() => {
    audioCaptureRef.current?.stop();
    audioCaptureRef.current = null;
    audioWsRef.current?.disconnect();
    audioWsRef.current = null;
    setRecording(false);
    setAudioConnected(false);
  }, [setRecording, setAudioConnected]);

  return (
    <div className="flex h-screen bg-zinc-950 text-white">
      {/* Sidebar */}
      <div
        className={cn(
          "shrink-0 border-r border-white/[0.06] bg-zinc-900/30 flex flex-col",
          "transition-[width] duration-200 ease-in-out overflow-hidden",
          showSidebar ? "w-60" : "w-0",
        )}
      >
        <SessionList />
      </div>

      {/* Main content */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-white/[0.06] px-4 py-2" style={{ WebkitAppRegion: "drag" } as React.CSSProperties}>
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setShowSidebar((v) => !v)}
              className="rounded-lg p-1 hover:bg-white/5 transition-colors"
              title="Toggle sidebar (Ctrl+B)"
            >
              <Logo size={20} />
            </button>
            {currentSession && (
              editingHeaderName ? (
                <input
                  autoFocus
                  value={headerNameValue}
                  onChange={(e) => setHeaderNameValue(e.target.value)}
                  onKeyDown={async (e) => {
                    if (e.key === "Enter") {
                      const trimmed = headerNameValue.trim();
                      if (trimmed && currentSession) {
                        try {
                          await renameSession(currentSession.id, trimmed);
                          renameCurrentSession(trimmed);
                        } catch (err) {
                          console.error("Failed to rename session:", err);
                        }
                      }
                      setEditingHeaderName(false);
                    }
                    if (e.key === "Escape") setEditingHeaderName(false);
                  }}
                  onBlur={async () => {
                    const trimmed = headerNameValue.trim();
                    if (trimmed && currentSession) {
                      try {
                        await renameSession(currentSession.id, trimmed);
                        renameCurrentSession(trimmed);
                      } catch (err) {
                        console.error("Failed to rename session:", err);
                      }
                    }
                    setEditingHeaderName(false);
                  }}
                  className="text-sm text-white/80 bg-transparent border-b border-white/20 outline-none font-medium"
                />
              ) : (
                <button
                  onClick={() => {
                    setEditingHeaderName(true);
                    setHeaderNameValue(currentSession.name);
                  }}
                  className="text-sm text-white/60 hover:text-white/90 transition-colors font-medium"
                  title="Click to rename session"
                >
                  {currentSession.name}
                </button>
              )
            )}
          </div>

          <div className="flex items-center gap-1">
            {currentSession && (
              <SessionContextInput onContextChange={handleSessionContextChange} />
            )}

            {currentSession && (
              <button
                onClick={recording ? stopRecording : startRecording}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
                  recording
                    ? "bg-red-500/15 text-red-400 hover:bg-red-500/25 border border-red-500/20"
                    : "bg-blue-500/15 text-blue-400 hover:bg-blue-500/25 border border-blue-500/20",
                )}
              >
                {recording ? (
                  <>
                    <Square className="h-3 w-3" /> Stop
                  </>
                ) : (
                  <>
                    <Mic className="h-3 w-3" /> Record
                  </>
                )}
              </button>
            )}

            <HeaderMoreMenu />

            <button
              onClick={() => setShowSettings(true)}
              className="rounded-lg p-1.5 text-white/40 hover:text-white/80 hover:bg-white/5 transition-colors"
            >
              <Settings className="h-4 w-4" />
            </button>

            <WindowControls />
          </div>
        </header>

        {/* Content area */}
        {currentSession ? (
          <div className="flex flex-1 min-h-0">
            <div className="flex-1 min-w-0">
              <TranscriptPanel />
            </div>

            {!insightsDrawerOpen && <InsightsPill />}

            <InsightsDrawer />
          </div>
        ) : !serverOnline ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <ServerOff className="h-8 w-8 text-white/15 mx-auto mb-4" />
              <h2 className="text-base font-semibold text-white/50 mb-2">Server Unavailable</h2>
              <p className="text-sm text-white/30 mb-4">
                Waiting for server at <span className="text-white/50">{serverUrl}</span>
              </p>
              <Loader2 className="h-4 w-4 text-white/20 mx-auto animate-spin" />
            </div>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center max-w-sm">
              <Logo size={40} className="mx-auto mb-4 opacity-20" />
              <h2 className="text-base font-semibold text-white/50 mb-2">Start a Conversation</h2>
              <p className="text-sm text-white/30 mb-6">
                Create a new session to begin capturing and analyzing conversations in real time.
              </p>
              <button
                onClick={() => {
                  if (!showSidebar) setShowSidebar(true);
                }}
                className="inline-flex items-center gap-2 rounded-lg bg-blue-500/15 border border-blue-500/20 px-4 py-2 text-sm text-blue-400 hover:bg-blue-500/25 transition-colors"
              >
                <Plus className="h-4 w-4" />
                New Session
              </button>
            </div>
          </div>
        )}

        <StatusBar />
      </div>

      {showSettings && <SettingsView onClose={() => setShowSettings(false)} />}
    </div>
  );
}

export default function App() {
  if (isOverlayRoute()) {
    return <OverlayHUD />;
  }
  return <MainApp />;
}
