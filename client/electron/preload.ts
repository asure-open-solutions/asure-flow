import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  toggleOverlay: () => ipcRenderer.invoke("toggle-overlay"),
  getAudioSources: () => ipcRenderer.invoke("get-audio-sources"),
  getEnvServerUrl: () => process.env.ASUREFLOW_SERVER || null,
  setIgnoreMouseEvents: (ignore: boolean, forward: boolean) =>
    ipcRenderer.send("set-ignore-mouse-events", ignore, { forward }),
  setContentProtection: (enabled: boolean) =>
    ipcRenderer.invoke("set-content-protection", enabled),

  setOverlayBounds: (bounds: { x?: number; y?: number; width?: number; height?: number }) =>
    ipcRenderer.invoke("set-overlay-bounds", bounds),

  getScreenSize: () => ipcRenderer.invoke("get-screen-size"),
  getPlatform: () => ipcRenderer.invoke("get-platform"),

  // Window controls
  windowMinimize: () => ipcRenderer.send("window-minimize"),
  windowMaximize: () => ipcRenderer.send("window-maximize"),
  windowClose: () => ipcRenderer.send("window-close"),
  windowIsMaximized: () => ipcRenderer.invoke("window-is-maximized"),
  onMaximizeChange: (callback: (maximized: boolean) => void) => {
    const handler = (_event: unknown, maximized: boolean) => callback(maximized);
    ipcRenderer.on("maximize-change", handler);
    return () => {
      ipcRenderer.removeListener("maximize-change", handler);
    };
  },

  // Overlay state sync (main window → overlay window via main process relay)
  sendOverlaySync: (data: unknown) => ipcRenderer.send("overlay-sync", data),
  onOverlaySync: (callback: (data: unknown) => void) => {
    const handler = (_event: unknown, data: unknown) => callback(data);
    ipcRenderer.on("overlay-sync", handler);
    return () => { ipcRenderer.removeListener("overlay-sync", handler); };
  },
  onOverlayOpened: (callback: () => void) => {
    ipcRenderer.on("overlay-opened", callback);
    return () => { ipcRenderer.removeListener("overlay-opened", callback); };
  },

  // Listen for events from main process
  onOverlayToggle: (callback: (isOverlay: boolean) => void) => {
    const handler = (_event: unknown, value: boolean) => callback(value);
    ipcRenderer.on("overlay-toggled", handler);
    return () => {
      ipcRenderer.removeListener("overlay-toggled", handler);
    };
  },

  // Recording toggle (overlay ↔ main window via main process relay)
  toggleRecording: () => ipcRenderer.send("toggle-recording"),
  onToggleRecording: (callback: () => void) => {
    const handler = () => callback();
    ipcRenderer.on("toggle-recording", handler);
    return () => { ipcRenderer.removeListener("toggle-recording", handler); };
  },

  // Audio toggle (overlay → main window via main process relay)
  setAudioToggle: (toggle: { mic?: boolean; system?: boolean }) =>
    ipcRenderer.send("set-audio-toggle", toggle),
  onAudioToggle: (callback: (toggle: { mic?: boolean; system?: boolean }) => void) => {
    const handler = (_event: unknown, toggle: { mic?: boolean; system?: boolean }) => callback(toggle);
    ipcRenderer.on("set-audio-toggle", handler);
    return () => { ipcRenderer.removeListener("set-audio-toggle", handler); };
  },
});
