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
import { setServerUrl, getServerConfig, checkServerHealth, renameSession, createSession, listSessions } from "@/services/api";
import { cn } from "@/lib/utils";
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
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
  const setAudioWarning = useSessionStore((s) => s.setAudioWarning);
  const setAudioConnected = useSessionStore((s) => s.setAudioConnected);
  const setSessionConnected = useSessionStore((s) => s.setSessionConnected);
  const addTranscriptEntry = useSessionStore((s) => s.addTranscriptEntry);
  const relabelSpeaker = useSessionStore((s) => s.relabelSpeaker);
  const handleAIEvent = useSessionStore((s) => s.handleAIEvent);
  const clearAgentLog = useSessionStore((s) => s.clearAgentLog);
  const insightsDrawerOpen = useSessionStore((s) => s.insightsDrawerOpen);
  const setInsightsDrawerOpen = useSessionStore((s) => s.setInsightsDrawerOpen);
  const setCurrentSession = useSessionStore((s) => s.setCurrentSession);
  const setSessions = useSessionStore((s) => s.setSessions);
  const rerunRequested = useSessionStore((s) => s.rerunRequested);
  const clearRerunRequest = useSessionStore((s) => s.clearRerunRequest);

  const serverUrl = useSettingsStore((s) => s.serverUrl);
  const hydrated = useSettingsStore((s) => s._hydrated);
  const effectiveToggles = useSettingsStore((s) => s.getEffectiveToggles());
  const audioToggles = useSettingsStore((s) => s.audioToggles);
  const setAudioToggles = useSettingsStore((s) => s.setAudioToggles);
  const contentProtection = useSettingsStore((s) => s.overlaySettings.contentProtection);
  const setLlmStatus = useSessionStore((s) => s.setLlmStatus);

  const audioWsRef = useRef<AudioWebSocket | null>(null);
  const sessionWsRef = useRef<SessionWebSocket | null>(null);
  const audioCaptureRef = useRef<AudioCapture | null>(null);
  const serverHandlesSystemRef = useRef(false);
  const serverConfigLoaded = useRef(false);
  const startingRecordingRef = useRef(false);

  // Keep server URL in sync — wait for hydration to avoid overwriting with default
  useEffect(() => {
    if (hydrated) setServerUrl(serverUrl);
  }, [serverUrl, hydrated]);

  // Poll server health — wait for hydration so we use the correct server URL
  useEffect(() => {
    if (!hydrated) return;
    let active = true;
    const check = async () => {
      const health = await checkServerHealth();
      if (active) {
        setServerOnline(health.online);
        setLlmStatus(health.llmAvailable, health.llmProvider);
      }
    };
    check();
    const ms = serverOnline ? 30_000 : 3_000;
    const id = setInterval(check, ms);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [serverOnline, serverUrl, hydrated]);

  // Load server config once on first successful connection to sync server-authoritative settings
  useEffect(() => {
    if (serverOnline && !serverConfigLoaded.current) {
      serverConfigLoaded.current = true;
      getServerConfig()
        .then((cfg) => useSettingsStore.getState().initFromServerConfig(cfg))
        .catch(() => {});
    }
  }, [serverOnline]);

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
    sws.onConnectionChange = (connected) => {
      setSessionConnected(connected);
      // On (re)connect, send config and rerun agent if transcript exists
      if (connected) {
        sws.sendConfig(useSettingsStore.getState().getEffectiveToggles());
        const { transcript } = useSessionStore.getState();
        if (transcript.length > 0) {
          sws.sendRerun();
        }
      }
    };
    sws.connect();
    sessionWsRef.current = sws;

    return () => {
      sws.disconnect();
    };
  }, [currentSession?.id]);

  // Sync effective feature toggles — rerun agent if a feature was enabled
  const prevTogglesRef = useRef(effectiveToggles);
  useEffect(() => {
    sessionWsRef.current?.sendConfig(effectiveToggles);

    // Detect if any boolean toggle went from false→true (feature enabled)
    const prev = prevTogglesRef.current;
    const boolKeys = Object.keys(prev).filter((k) => k !== "deep_think") as (keyof Omit<typeof prev, "deep_think">)[];
    const featureEnabled = boolKeys.some((k) => !prev[k] && effectiveToggles[k])
      || (prev.deep_think === "off" && effectiveToggles.deep_think !== "off");
    prevTogglesRef.current = effectiveToggles;

    if (featureEnabled) {
      const { transcript } = useSessionStore.getState();
      if (transcript.length > 0) {
        sessionWsRef.current?.sendRerun();
      }
    }
  }, [effectiveToggles]);

  // Watch rerun flag from store (set by TranscriptPanel after edit/delete)
  useEffect(() => {
    if (rerunRequested) {
      sessionWsRef.current?.sendRerun();
      clearRerunRequest();
    }
  }, [rerunRequested, clearRerunRequest]);

  // Sync audio toggles while recording
  useEffect(() => {
    const capture = audioCaptureRef.current;
    if (!capture || !recording) return;

    if (audioToggles.mic) {
      capture.enableMic().catch((err) => {
        const msg = err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone permission denied"
          : `Microphone error: ${(err as Error).message}`;
        setAudioWarning(msg);
      });
    } else {
      capture.disableMic();
    }

    // Skip client-side system capture when server handles it via loopback
    if (!serverHandlesSystemRef.current) {
      if (audioToggles.system) {
        capture.enableSystem().catch((err) => {
          const msg = err instanceof DOMException && err.name === "NotAllowedError"
            ? "System audio permission denied"
            : `System audio error: ${(err as Error).message}`;
          setAudioWarning(msg);
        });
      } else {
        capture.disableSystem();
      }
    }
  }, [audioToggles.mic, audioToggles.system, recording]);

  // Sync content protection
  useEffect(() => {
    window.electronAPI?.setContentProtection(contentProtection);
  }, [contentProtection]);

  // Sync session state to overlay window via IPC (debounced to avoid flooding during AI streaming)
  useEffect(() => {
    const sendSync = () => {
      const { transcript, suggestions, notes, recording: rec, recordingStartedAt: rsa } = useSessionStore.getState();
      const latestSuggestion = suggestions[suggestions.length - 1]?.text ?? null;
      window.electronAPI?.sendOverlaySync({
        transcript: transcript.slice(-20),
        latestSuggestion,
        notes,
        recording: rec,
        recordingStartedAt: rsa,
        audioToggles: useSettingsStore.getState().audioToggles,
      });
    };

    let syncTimer: ReturnType<typeof setTimeout> | null = null;
    const debouncedSync = () => {
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(sendSync, 200);
    };

    // Subscribe to store changes — relay to overlay (debounced)
    const unsub = useSessionStore.subscribe(debouncedSync);

    // When overlay opens, send current state immediately (not debounced)
    const cleanupOpened = window.electronAPI?.onOverlayOpened(sendSync);

    return () => {
      if (syncTimer) clearTimeout(syncTimer);
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
        setInsightsDrawerOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // IPC: overlay requests recording toggle
  const startRecordingRef = useRef<() => void>(undefined);
  const stopRecordingRef = useRef<() => void>(undefined);
  useEffect(() => {
    const cleanup = window.electronAPI?.onToggleRecording(() => {
      const store = useSessionStore.getState();
      if (store.recording) {
        stopRecordingRef.current?.();
      } else if (store.currentSession) {
        startRecordingRef.current?.();
      }
    });
    return () => cleanup?.();
  }, []);

  // IPC: overlay requests audio toggle change
  useEffect(() => {
    const cleanup = window.electronAPI?.onAudioToggle((toggle) => {
      useSettingsStore.getState().setAudioToggles(toggle);
    });
    return () => cleanup?.();
  }, []);

  const handleSessionContextChange = useCallback(
    (context: string) => {
      sessionWsRef.current?.sendSessionContext(context);
      // Rerun agent with new context if transcript exists
      const { transcript } = useSessionStore.getState();
      if (transcript.length > 0) {
        sessionWsRef.current?.sendRerun();
      }
    },
    [],
  );

  const startRecording = useCallback(async () => {
    if (recording || startingRecordingRef.current) return;
    startingRecordingRef.current = true;

    let micDeviceId: string | undefined;
    let serverHandlesSystem = false;
    try {
      const cfg = await getServerConfig();
      if (cfg.audio_capture_source === "client") {
        // Use the client-local device preference (not the server-stored one,
        // which may refer to hardware on a different machine)
        micDeviceId = useSettingsStore.getState().micDeviceId ?? undefined;
      }
      // If system_device_id is set, server handles system audio via loopback
      serverHandlesSystem = !!cfg.system_device_id;
    } catch {
      // Proceed with system default
    }
    serverHandlesSystemRef.current = serverHandlesSystem;

    // Start audio capture first — check permissions before connecting WebSocket
    const capture = new AudioCapture();
    const wantMic = audioToggles.mic;
    const wantSystem = audioToggles.system && !serverHandlesSystem;

    const result = await capture.start({
      mic: wantMic,
      system: wantSystem,
      micDeviceId,
    });

    // If nothing is capturing (and server isn't handling system), bail out
    const serverHandlesEverything = serverHandlesSystem && !wantMic;
    if (!result.mic && !result.system && !serverHandlesEverything) {
      capture.stop();
      startingRecordingRef.current = false;
      const errors = [result.micError, result.systemError].filter(Boolean).join(". ");
      setAudioWarning(errors || "No audio sources available");
      return;
    }

    // Build a warning for partial captures
    const warnings: string[] = [];
    if (wantMic && !result.mic && result.micError) warnings.push(result.micError);
    if (wantSystem && !result.system && result.systemError) warnings.push(result.systemError);
    setAudioWarning(warnings.length > 0 ? warnings.join(". ") : null);

    // Audio is capturing — now connect the WebSocket
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

    capture.onChunk = (streamId, pcmData) => {
      aws.sendAudio(streamId, pcmData);
    };
    audioCaptureRef.current = capture;

    startingRecordingRef.current = false;
    setRecording(true);
  }, [recording, audioToggles, addTranscriptEntry, relabelSpeaker, setAudioConnected, setAudioWarning, setRecording]);

  const stopRecording = useCallback(() => {
    audioCaptureRef.current?.stop();
    audioCaptureRef.current = null;
    audioWsRef.current?.disconnect();
    audioWsRef.current = null;
    setRecording(false);
    setAudioConnected(false);
  }, [setRecording, setAudioConnected]);

  // Keep refs in sync for IPC callbacks (which can't capture latest closures)
  startRecordingRef.current = startRecording;
  stopRecordingRef.current = stopRecording;

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
              <div className="flex items-center gap-0.5">
                <button
                  onClick={() => setAudioToggles({ mic: !audioToggles.mic })}
                  className={cn(
                    "rounded-lg p-1.5 transition-colors",
                    audioToggles.mic
                      ? "text-white/40 hover:text-white/70 hover:bg-white/5"
                      : "text-red-400/60 hover:text-red-400 hover:bg-red-500/10",
                  )}
                  title={audioToggles.mic ? "Mute microphone" : "Unmute microphone"}
                  style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
                >
                  {audioToggles.mic ? <Mic className="h-3.5 w-3.5" /> : <MicOff className="h-3.5 w-3.5" />}
                </button>
                <button
                  onClick={() => setAudioToggles({ system: !audioToggles.system })}
                  className={cn(
                    "rounded-lg p-1.5 transition-colors",
                    audioToggles.system
                      ? "text-white/40 hover:text-white/70 hover:bg-white/5"
                      : "text-red-400/60 hover:text-red-400 hover:bg-red-500/10",
                  )}
                  title={audioToggles.system ? "Mute system audio" : "Unmute system audio"}
                  style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
                >
                  {audioToggles.system ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
                </button>
              </div>
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
                onClick={async () => {
                  try {
                    const session = await createSession();
                    setCurrentSession(session);
                    const list = await listSessions();
                    setSessions(list);
                    if (!showSidebar) setShowSidebar(true);
                  } catch (err) {
                    console.error("Failed to create session:", err);
                  }
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
