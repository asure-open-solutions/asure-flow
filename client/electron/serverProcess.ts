/**
 * Manages the bundled Python server lifecycle for the packaged one-click app.
 *
 * In a packaged build the FastAPI server ships as a relocatable Python runtime
 * under `resources/server` (produced by scripts/build-server-bundle.mjs and copied
 * in via electron-builder `extraResources`). Electron spawns it on startup and
 * tears it down on quit, so the user never installs Python or runs a script.
 *
 * In development the server is started separately (scripts/launch.mjs), so
 * startServer() is a no-op there unless a venv interpreter happens to exist.
 *
 * See PACKAGING.md for the full design and the two wiring edits in main.ts /
 * package.json that connect this module.
 */
import { spawn, ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import http from "node:http";
import { app } from "electron";

const PORT = 8000;
const HOST = "127.0.0.1";

let serverProcess: ChildProcess | null = null;

interface ServerLaunch {
  python: string;
  cwd: string;
  args: string[];
}

/** Locate the Python interpreter + working dir, or null if no server is bundled. */
function resolveServer(): ServerLaunch | null {
  const isWindows = process.platform === "win32";
  const args = ["-m", "uvicorn", "asure_flow.main:app", "--host", HOST, "--port", String(PORT)];

  if (app.isPackaged) {
    // Relocatable runtime laid down by build-server-bundle.mjs.
    const base = path.join(process.resourcesPath, "server");
    const python = isWindows
      ? path.join(base, "runtime", "python.exe")
      : path.join(base, "runtime", "bin", "python3");
    return existsSync(python) ? { python, cwd: base, args } : null;
  }

  // Dev: reuse the project venv if present (otherwise launch.mjs handles it).
  const root = path.resolve(__dirname, "..", "..");
  const python = isWindows
    ? path.join(root, "server", ".venv", "Scripts", "python.exe")
    : path.join(root, "server", ".venv", "bin", "python");
  return existsSync(python) ? { python, cwd: path.join(root, "server"), args } : null;
}

/** Resolve true once the server answers /api/health. */
function ping(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`http://${HOST}:${PORT}/api/health`, { timeout: 2000 }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

/**
 * Start the bundled server. Returns false if nothing was bundled (caller should
 * fall back to whatever server URL the user has configured). Only starts in
 * packaged builds — in dev, launch.mjs owns the server.
 */
export function startServer(onLog?: (line: string) => void): boolean {
  if (!app.isPackaged) return false;
  if (serverProcess) return true;

  const launch = resolveServer();
  if (!launch) {
    onLog?.("[server] No bundled runtime found — expecting an external server.");
    return false;
  }

  serverProcess = spawn(launch.python, launch.args, {
    cwd: launch.cwd,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    windowsHide: true,
  });

  serverProcess.stdout?.on("data", (d: Buffer) => onLog?.(`[server] ${d.toString().trimEnd()}`));
  serverProcess.stderr?.on("data", (d: Buffer) => onLog?.(`[server] ${d.toString().trimEnd()}`));
  serverProcess.on("exit", (code) => {
    onLog?.(`[server] exited with code ${code}`);
    serverProcess = null;
  });

  return true;
}

/** Resolve once the server is reachable, or false after timeoutMs. */
export async function waitForHealth(timeoutMs = 180_000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await ping()) return true;
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

/** Stop the server process (and its child tree on Windows). Best-effort. */
export function stopServer(): void {
  if (!serverProcess?.pid) return;
  const pid = serverProcess.pid;
  if (process.platform === "win32") {
    // Python child trees don't reliably die on a plain kill() on Windows.
    try {
      spawn("taskkill", ["/pid", String(pid), "/T", "/F"], { windowsHide: true });
    } catch {
      serverProcess.kill();
    }
  } else {
    serverProcess.kill("SIGTERM");
  }
  serverProcess = null;
}
