import {
  app,
  BrowserWindow,
  desktopCapturer,
  ipcMain,
  screen,
  Tray,
  Menu,
  nativeImage,
  globalShortcut,
} from "electron";
import path from "node:path";
import { createOverlayWindow } from "./overlay";
import { applyContentProtection } from "./contentProtection";
import { createTrayIcon, createAppIcon } from "./trayIcon";

let mainWindow: BrowserWindow | null = null;
let overlayWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let isOverlayMode = false;
let contentProtectionEnabled = true;

const DIST = path.join(__dirname, "../dist");
const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL;

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: "Asuré Flow",
    icon: createAppIcon(),
    frame: false,
    autoHideMenuBar: true,
    show: false,
    backgroundColor: "#09090b",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
  });

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(DIST, "index.html"));
  }

  applyContentProtection(mainWindow, contentProtectionEnabled);

  // Bypass screen-share dialog — auto-select first screen with loopback audio.
  // This makes getDisplayMedia() in the renderer work without showing a picker.
  mainWindow.webContents.session.setDisplayMediaRequestHandler(
    async (_request, callback) => {
      const sources = await desktopCapturer.getSources({ types: ["screen"] });
      callback({ video: sources[0], audio: "loopback" });
    },
  );

  // Forward maximize state changes to renderer for title bar icon updates
  mainWindow.on("maximize", () => mainWindow?.webContents.send("maximize-change", true));
  mainWindow.on("unmaximize", () => mainWindow?.webContents.send("maximize-change", false));

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function createTray() {
  const icon = createTrayIcon();
  tray = new Tray(icon);
  tray.setToolTip("Asuré Flow");

  const contextMenu = Menu.buildFromTemplate([
    {
      label: "Show Main Window",
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        }
      },
    },
    {
      label: "Toggle Overlay",
      click: () => toggleOverlay(),
    },
    { type: "separator" },
    {
      label: "Quit",
      click: () => {
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(contextMenu);
}

function toggleOverlay() {
  if (isOverlayMode) {
    // Close overlay, show main window
    overlayWindow?.close();
    overlayWindow = null;
    if (overlayThrottle) { clearTimeout(overlayThrottle); overlayThrottle = null; }
    pendingOverlayData = null;
    mainWindow?.show();
    isOverlayMode = false;
  } else {
    // Create overlay, hide main window
    overlayWindow = createOverlayWindow(VITE_DEV_SERVER_URL, DIST, contentProtectionEnabled);
    mainWindow?.hide();
    isOverlayMode = true;

    // Ask main window to send current state so overlay gets initial data
    overlayWindow.webContents.once("did-finish-load", () => {
      mainWindow?.webContents.send("overlay-opened");
    });

    overlayWindow.on("closed", () => {
      overlayWindow = null;
      isOverlayMode = false;
      mainWindow?.show();
    });
  }
}

// ── IPC Handlers ──

// Relay session state from main window to overlay window (throttled)
let overlayThrottle: ReturnType<typeof setTimeout> | null = null;
let pendingOverlayData: unknown = null;

ipcMain.on("overlay-sync", (_event, data) => {
  pendingOverlayData = data;
  if (!overlayThrottle) {
    overlayWindow?.webContents.send("overlay-sync", data);
    overlayThrottle = setTimeout(() => {
      overlayThrottle = null;
      if (pendingOverlayData !== null) {
        overlayWindow?.webContents.send("overlay-sync", pendingOverlayData);
        pendingOverlayData = null;
      }
    }, 100);
  }
});

ipcMain.on("set-ignore-mouse-events", (_event, ignore: boolean, options?: { forward: boolean }) => {
  if (overlayWindow) {
    if (process.platform === "linux") {
      // On Linux, the forward option is not supported — toggle between
      // fully click-through and fully interactive instead.
      overlayWindow.setIgnoreMouseEvents(ignore);
    } else {
      overlayWindow.setIgnoreMouseEvents(ignore, options);
    }
  }
});

ipcMain.handle("toggle-overlay", () => {
  toggleOverlay();
  return isOverlayMode;
});

ipcMain.handle("set-content-protection", (_event, enabled: boolean) => {
  contentProtectionEnabled = enabled;
  if (mainWindow) {
    applyContentProtection(mainWindow, enabled);
  }
  if (overlayWindow) {
    applyContentProtection(overlayWindow, enabled);
  }
});

ipcMain.handle("get-audio-sources", async () => {
  const sources = await desktopCapturer.getSources({ types: ["screen"] });
  return sources.map((s) => ({ id: s.id, name: s.name }));
});

ipcMain.handle("set-overlay-bounds", (_event, bounds: { x?: number; y?: number; width?: number; height?: number }) => {
  if (overlayWindow) {
    const current = overlayWindow.getBounds();
    overlayWindow.setBounds({
      x: bounds.x ?? current.x,
      y: bounds.y ?? current.y,
      width: bounds.width ?? current.width,
      height: bounds.height ?? current.height,
    });
  }
});

ipcMain.handle("get-screen-size", () => {
  return screen.getPrimaryDisplay().workAreaSize;
});

ipcMain.handle("get-platform", () => process.platform);

// ── Recording + Audio Toggle Relay (overlay ↔ main window) ──

ipcMain.on("toggle-recording", () => {
  mainWindow?.webContents.send("toggle-recording");
});

ipcMain.on("set-audio-toggle", (_event, toggle: { mic?: boolean; system?: boolean }) => {
  mainWindow?.webContents.send("set-audio-toggle", toggle);
});

// ── Window Controls ──

ipcMain.on("window-minimize", () => mainWindow?.minimize());
ipcMain.on("window-maximize", () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
});
ipcMain.on("window-close", () => mainWindow?.close());
ipcMain.handle("window-is-maximized", () => mainWindow?.isMaximized() ?? false);

// ── App Lifecycle ──

app.whenReady().then(() => {
  createMainWindow();
  createTray();

  // Global shortcut to toggle overlay
  globalShortcut.register("CommandOrControl+Shift+O", toggleOverlay);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (!mainWindow) {
    createMainWindow();
  }
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});
