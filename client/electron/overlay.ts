import { BrowserWindow, screen } from "electron";
import path from "node:path";
import { applyContentProtection } from "./contentProtection";

export function createOverlayWindow(
  devServerUrl: string | undefined,
  distPath: string,
  contentProtection: boolean = true,
): BrowserWindow {
  const { width: screenW, height: screenH } = screen.getPrimaryDisplay().workAreaSize;

  // Full-screen transparent canvas — React components position themselves within it.
  // This avoids needing to recreate the window when switching overlay modes.
  const overlay = new BrowserWindow({
    width: screenW,
    height: screenH,
    x: 0,
    y: 0,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    resizable: false,
    // On Linux, keep focusable so the overlay can receive mouse events
    // since setIgnoreMouseEvents' forward option is not supported.
    focusable: process.platform === "linux",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  overlay.setAlwaysOnTop(true, "screen-saver");
  overlay.setVisibleOnAllWorkspaces(true);
  applyContentProtection(overlay, contentProtection);

  // On Linux, setIgnoreMouseEvents with { forward: true } is not supported.
  // The overlay starts interactive; the renderer handles click-through via
  // CSS pointer-events: none on transparent regions.
  if (process.platform !== "linux") {
    overlay.setIgnoreMouseEvents(true, { forward: true });
  }

  if (devServerUrl) {
    overlay.loadURL(`${devServerUrl}#/overlay`);
  } else {
    overlay.loadFile(path.join(distPath, "index.html"), { hash: "/overlay" });
  }

  return overlay;
}
