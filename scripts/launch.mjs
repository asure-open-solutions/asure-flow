#!/usr/bin/env node
/**
 * Cross-platform launcher for AsuréFlow.
 * Usage: node scripts/launch.mjs [target]
 *   target: "server" | "client" | "setup" | (default) both server + client
 */
import { spawn } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const isWin = process.platform === "win32";
const target = process.argv[2] || "all";

const scripts = {
  all:    isWin ? "start.bat"        : "start.sh",
  server: isWin ? "start-server.bat" : "start-server.sh",
  client: isWin ? "start-client.bat" : "start-client.sh",
  setup:  isWin ? "scripts\\setup.ps1" : "scripts/setup.sh",
};

const script = scripts[target];
if (!script) {
  console.error(`Unknown target: ${target}`);
  console.error("Usage: node scripts/launch.mjs [server|client|setup]");
  process.exit(1);
}

const file = join(root, script);

let cmd, args;
if (isWin) {
  if (script.endsWith(".ps1")) {
    cmd = "powershell";
    args = ["-ExecutionPolicy", "Bypass", "-File", file];
  } else {
    cmd = "cmd";
    args = ["/c", file];
  }
} else {
  cmd = "bash";
  args = [file];
}

const child = spawn(cmd, args, {
  stdio: "inherit",
  cwd: root,
});

const forward = (signal) => {
  child.kill(signal);
};
process.on("SIGINT", () => forward("SIGINT"));
process.on("SIGTERM", () => forward("SIGTERM"));

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
