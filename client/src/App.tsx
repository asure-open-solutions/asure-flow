import { Component, useCallback, useEffect, useRef, useState } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useShallow } from "zustand/react/shallow";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import { SessionList } from "@/components/SessionList";
import { SettingsView } from "@/components/SettingsView";
import { StatusBar } from "@/components/StatusBar";
import { SessionContextInput } from "@/components/SessionContextInput";
import { HeaderMoreMenu } from "@/components/HeaderMoreMenu";
import { InsightsDrawer } from "@/components/InsightsDrawer";
import { InsightsPill } from "@/components/InsightsPill";
import { HomePage } from "@/components/HomePage";
import { OverlayHUD } from "@/components/overlay/OverlayHUD";
import { WindowControls } from "@/components/WindowControls";
import { AudioCapture } from "@/services/audioCapture";
import { AudioWebSocket, SessionWebSocket } from "@/services/websocket";
import { setServerUrl, getServerConfig, getProfile, checkServerHealth, renameSession, createSession, listSessions, getSession } from "@/services/api";
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
  // Session data (re-render-relevant, shallow-compared)
  const {
    currentSession, recording, serverOnline, sessions,
    insightsDrawerOpen, rerunRequested,
  } = useSessionStore(useShallow((s) => ({
    currentSession: s.currentSession,
    recording: s.recording,
    serverOnline: s.serverOnline,
    sessions: s.sessions,
    insightsDrawerOpen: s.insightsDrawerOpen,
    rerunRequested: s.rerunRequested,
  })));

  // Session actions (stable function refs — single subscription)
  const {
    renameCurrentSession, setServerOnline, setRecording, setAudioWarning,
    setAudioConnected, setSessionConnected, addTranscriptEntry, relabelSpeaker,
    handleAIEvent, clearAgentLog, setInsightsDrawerOpen, setCurrentSession,
    setSessions, clearRerunRequest, setLlmStatus,
  } = useSessionStore(useShallow((s) => ({
    renameCurrentSession: s.renameCurrentSession,
    setServerOnline: s.setServerOnline,
    setRecording: s.setRecording,
    setAudioWarning: s.setAudioWarning,
    setAudioConnected: s.setAudioConnected,
    setSessionConnected: s.setSessionConnected,
    addTranscriptEntry: s.addTranscriptEntry,
    relabelSpeaker: s.relabelSpeaker,
    handleAIEvent: s.handleAIEvent,
    clearAgentLog: s.clearAgentLog,
    setInsightsDrawerOpen: s.setInsightsDrawerOpen,
    setCurrentSession: s.setCurrentSession,
    setSessions: s.setSessions,
    clearRerunRequest: s.clearRerunRequest,
    setLlmStatus: s.setLlmStatus,
  })));

  // Settings data (shallow-compared)
  const { serverUrl, hydrated, audioToggles, contentProtection } = useSettingsStore(
    useShallow((s) => ({
      serverUrl: s.serverUrl,
      hydrated: s._hydrated,
      audioToggles: s.audioToggles,
      contentProtection: s.overlaySettings.contentProtection,
    })),
  );
  const effectiveToggles = useSettingsStore(useShallow((s) => s.getEffectiveToggles()));
  const setAudioToggles = useSettingsStore((s) => s.setAudioToggles);

  const audioWsRef = useRef<AudioWebSocket | null>(null);
  const sessionWsRef = useRef<SessionWebSocket | null>(null);
  const audioCaptureRef = useRef<AudioCapture | null>(null);
  const serverHandlesSystemRef = useRef(false);
  const serverConfigLoaded = useRef(false);
  const startingRecordingRef = useRef(false);

  // Keep server URL in sync — wait for hydration to avoid overwriting with default
  useEffect(() => {
    if (hydrated) setServerUrl(serverUrl);
    // Reset so config/profile reload from the new server
    serverConfigLoaded.current = false;
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

  // Load server config + user profile once on first successful connection (or after URL change)
  useEffect(() => {
    if (serverOnline && !serverConfigLoaded.current) {
      serverConfigLoaded.current = true;
      // Server config: admin/hardware settings (whisper model, providers, etc.)
      getServerConfig()
        .then((cfg) => useSettingsStore.getState().initFromServerConfig(cfg))
        .catch(() => {});
      // User profile: portable preferences (toggles, AI preset, privacy prefs)
      getProfile()
        .then((p) => useSettingsStore.getState().initFromServerProfile(p))
        .catch(() => {});
    }
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

  // Sync audio toggles while recording (serialised to avoid overlapping enable/disable)
  const audioToggleBusy = useRef(false);
  useEffect(() => {
    const capture = audioCaptureRef.current;
    if (!capture || !recording) return;
    if (audioToggleBusy.current) return;

    audioToggleBusy.current = true;
    (async () => {
      try {
        if (audioToggles.mic) {
          const deviceId = useSettingsStore.getState().micDeviceId ?? undefined;
          await capture.enableMic(deviceId);
        } else {
          capture.disableMic();
        }

        // Skip client-side system capture when server handles it via loopback
        if (!serverHandlesSystemRef.current) {
          if (audioToggles.system) {
            await capture.enableSystem();
          } else {
            capture.disableSystem();
          }
        }
      } catch (err) {
        const msg = err instanceof DOMException && err.name === "NotAllowedError"
          ? "Audio permission denied"
          : `Audio error: ${(err as Error).message}`;
        setAudioWarning(msg);
      } finally {
        audioToggleBusy.current = false;
      }
    })();
  }, [audioToggles.mic, audioToggles.system, recording]);

  // Sync content protection
  useEffect(() => {
    window.electronAPI?.setContentProtection(contentProtection);
  }, [contentProtection]);

  // Sync session state to overlay window via IPC (debounced to avoid flooding during AI streaming)
  useEffect(() => {
    const sendSync = () => {
      const { transcript, suggestions, focusedSuggestionId, notes, recording: rec, recordingStartedAt: rsa } = useSessionStore.getState();
      const latestSuggestion = suggestions[suggestions.length - 1]?.text ?? null;
      const focusedEntry = focusedSuggestionId ? suggestions.find((s) => s.id === focusedSuggestionId) : null;
      window.electronAPI?.sendOverlaySync({
        transcript: transcript.slice(-20),
        latestSuggestion,
        focusedSuggestionId,
        focusedSuggestionText: focusedEntry?.text ?? null,
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
        setInsightsDrawerOpen(!useSessionStore.getState().insightsDrawerOpen);
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

  // IPC: overlay requests suggestion focus change
  useEffect(() => {
    const cleanup = window.electronAPI?.onFocusSuggestion((id) => {
      useSessionStore.getState().focusSuggestion(id);
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

    let result;
    try {
      result = await capture.start({
        mic: wantMic,
        system: wantSystem,
        micDeviceId,
      });
    } catch (err) {
      capture.stop?.();
      startingRecordingRef.current = false;
      setAudioWarning(`Audio error: ${(err as Error).message}`);
      return;
    }

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
          <HomePage
            sessions={sessions}
            onCreateSession={async () => {
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
            onSelectSession={async (id) => {
              try {
                const session = await getSession(id);
                setCurrentSession(session);
              } catch (err) {
                console.error("Failed to load session:", err);
              }
            }}
          />
        )}

        <StatusBar />
      </div>

      {showSettings && <SettingsView onClose={() => setShowSettings(false)} />}
    </div>
  );
}

class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("App crashed:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex h-screen items-center justify-center bg-zinc-950 text-white p-8">
          <div className="max-w-lg text-center">
            <h1 className="text-lg font-semibold text-red-400 mb-2">Something went wrong</h1>
            <pre className="text-xs text-white/60 bg-white/5 rounded-lg p-4 text-left overflow-auto max-h-64 mb-4">
              {this.state.error.message}
              {"\n\n"}
              {this.state.error.stack}
            </pre>
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded-lg bg-blue-500/15 border border-blue-500/20 px-4 py-2 text-sm text-blue-400 hover:bg-blue-500/25"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  if (isOverlayRoute()) {
    return <OverlayHUD />;
  }
  return (
    <ErrorBoundary>
      <MainApp />
    </ErrorBoundary>
  );
}
