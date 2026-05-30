#!/usr/bin/env node
/**
 * Builds a self-contained Python server bundle for the one-click installer.
 *
 * Downloads a relocatable CPython (python-build-standalone) and installs the
 * AsuréFlow server + all deps into it, NON-editable so the package is copied in
 * and runs on a machine with no Python. electron-builder then ships
 * `client/server-bundle/` to `resources/server/` via `extraResources`.
 *
 * RUN PER TARGET PLATFORM (runtime + native wheels are OS/arch-specific):
 *   node scripts/build-server-bundle.mjs
 * then build the installer:  cd client && npm run build
 *
 * See PACKAGING.md for the rationale (why embedded portable Python, not freezing)
 * and known sharp edges (torch/CUDA, unsigned builds, first-run model download).
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, renameSync, createWriteStream } from "node:fs";
import { platform, arch } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "client", "server-bundle");
const runtimeDir = path.join(outDir, "runtime");

// Pin a python-build-standalone release. Bump deliberately; verify asset names.
const PBS_TAG = "20240814";
const PY_VERSION = "3.12.5";

function pbsAsset() {
  const a = arch(); // 'x64' | 'arm64'
  const p = platform(); // 'win32' | 'darwin' | 'linux'
  const triple = {
    "win32:x64": "x86_64-pc-windows-msvc-install_only",
    "darwin:x64": "x86_64-apple-darwin-install_only",
    "darwin:arm64": "aarch64-apple-darwin-install_only",
    "linux:x64": "x86_64-unknown-linux-gnu-install_only",
    "linux:arm64": "aarch64-unknown-linux-gnu-install_only",
  }[`${p}:${a}`];
  if (!triple) throw new Error(`Unsupported platform/arch: ${p}/${a}`);
  return `cpython-${PY_VERSION}+${PBS_TAG}-${triple}.tar.gz`;
}

function pythonExe() {
  return platform() === "win32"
    ? path.join(runtimeDir, "python.exe")
    : path.join(runtimeDir, "bin", "python3");
}

async function download(url, dest) {
  console.log(`downloading ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}): ${url}`);
  await pipeline(res.body, createWriteStream(dest));
}

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.status !== 0) throw new Error(`Command failed: ${cmd} ${args.join(" ")}`);
}

async function main() {
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });

  // 1. Fetch + extract the relocatable runtime.
  const asset = pbsAsset();
  const url = `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${asset}`;
  const tarball = path.join(outDir, asset);
  await download(url, tarball);

  console.log("extracting runtime");
  // install_only archives extract to a top-level `python/` dir.
  run("tar", ["-xzf", tarball, "-C", outDir]);
  renameSync(path.join(outDir, "python"), runtimeDir);
  rmSync(tarball, { force: true });

  const py = pythonExe();
  if (!existsSync(py)) throw new Error(`Runtime python missing after extract: ${py}`);

  // 2. Install the server into the runtime (non-editable -> copied in).
  run(py, ["-m", "pip", "install", "--upgrade", "pip"]);
  const torchIndex = process.env.TORCH_INDEX_URL;
  if (torchIndex) {
    console.log(`installing torch from ${torchIndex}`);
    run(py, ["-m", "pip", "install", "torch", "--index-url", torchIndex]);
  }
  const serverSpec = path.join(root, "server"); // installs from pyproject, non-editable
  run(py, ["-m", "pip", "install", serverSpec]);

  console.log(`\nServer bundle ready: ${outDir}`);
  console.log("Next: cd client && npm run build");
}

main().catch((e) => {
  console.error(`\nERROR: ${e.message}`);
  process.exit(1);
});
