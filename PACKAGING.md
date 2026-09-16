# Packaging AsuréFlow as a one-click app

**Goal:** a non-technical user double-clicks one installer, the app opens, and
everything (Python server + transcription + AI) runs — no Python, no Node, no
terminal, no editing `.env`.

> Status: **bundler + spawn module + docs committed; two wiring edits and the
> actual installer build remain.** The wiring edits are tiny and spelled out below
> under "Wiring". Building/verifying the real multi-GB installer is a manual,
> per-OS release step (checklist at the bottom) — it can't run in CI.

---

## Why embedded portable Python (not PyInstaller)

The server depends on heavy **native** ML packages: `faster-whisper`
(→ `ctranslate2`, C++), `torch`, `pyannote.audio`, `sentence-transformers`.

| Approach | Verdict for this stack |
|:---|:---|
| **Freeze** (PyInstaller / cx_Freeze) | Fragile here. torch + ctranslate2 + numpy need hand-tuned hidden-imports, bundled dynamic libs, and data-file hooks; every dep bump risks breakage. Rejected. |
| **Embed a portable Python** (chosen) | Ship a *relocatable* CPython and `pip install` the server into it from real PyPI wheels. No freezing — you run the same wheels that work in dev. Most robust for native deps. Larger on disk, which is fine. |

**Which portable Python:** [python-build-standalone] (PBS, by Astral — the `uv`
authors). Its `install_only` archives are explicitly **relocatable**, so the
runtime can live inside the app bundle and move to the user's machine.

**Relocation gotcha that drove the design:** a normal `venv` hardcodes absolute
paths and does **not** survive being copied to another machine. So we do **not**
ship a venv. `scripts/build-server-bundle.mjs` extracts PBS CPython and
`pip install <server>` directly into that interpreter's site-packages
(non-editable → fully copied in), then we invoke that interpreter directly.

**GPU vs CPU:** default to **CPU wheels** for distribution — GPU needs a matching
CUDA stack on the user's machine you can't guarantee. `ctranslate2` runs
`large-v3-turbo` at int8 on CPU acceptably, and the engine's OOM cascade already
prefers GPU when present and falls back to CPU int8 otherwise (now surfaced via
the "Transcription degraded" chip). A GPU build is opt-in via `TORCH_INDEX_URL`.

**Models are not bundled** (several GB). Whisper / pyannote /
sentence-transformers download on first use; the StatusBar shows server health
during first-run warm-up.

**API keys need no `.env`.** Configure providers in Settings → persisted to
`config.json`. `.env` stays an optional developer convenience.

---

## How it works at runtime

- In a **packaged** build, Electron's main process spawns the bundled server on
  startup (`electron/serverProcess.ts` → `startServer()`), pointing at
  `resources/server/runtime/python(.exe)` running `uvicorn asure_flow.main:app`
  on `127.0.0.1:8000`.
- The renderer **already** polls `/api/health` and shows "Offline" until the
  server answers — no extra splash logic needed.
- On quit, `stopServer()` tree-kills the process (`taskkill /T /F` on Windows,
  `SIGTERM` elsewhere).
- In **dev**, `scripts/launch.mjs` owns the server; `startServer()` returns early
  unless `app.isPackaged`.

---

## Wiring (two small edits to apply)

These touch existing files; left as documented steps because the dev shell was
too unstable to do exact-match edits safely in this pass.

**1. `client/electron/main.ts`**

```ts
// near the other imports:
import { startServer, stopServer } from "./serverProcess";

// inside app.whenReady().then(() => { ... }), before createWindow():
startServer((line) => console.log(line));

// in the existing before-quit handler (next to overlayManager.destroy()):
stopServer();
```

**2. `client/package.json`** — inside the existing `"build"` (electron-builder)
object, alongside `"asar": true`:

```jsonc
"extraResources": [
  { "from": "server-bundle", "to": "server", "filter": ["**/*"] }
]
```

Also add to `.gitignore`: `client/server-bundle/` (multi-GB build artifact;
`client/release/` is already ignored).

---

## Build steps (run on each target OS — native wheels are OS/arch-specific)

```bash
# 1. Build the portable server bundle (downloads PBS CPython + pip-installs the
#    server into client/server-bundle/). CPU build (default):
node scripts/build-server-bundle.mjs

#    GPU/CUDA build instead (optional, needs a CUDA box). bash example:
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 node scripts/build-server-bundle.mjs

# 2. Build the installer:
cd client
npm run build            # tsc -b && vite build && electron-builder
# → installer in client/release/
```

`npm run build:win` / `build:mac` / `build:linux` target a single platform.
electron-builder errors if `client/server-bundle/` is missing — always run step 1
first.

---

## Release checklist — manual verification (needs the target OS; not CI-able)

1. Run the produced installer on a **clean machine with no Python/Node**.
2. Launch. StatusBar should go Offline → Server within ~10–60 s (first run pulls
   models — watch the server log in the app's user-data dir).
3. Start recording; confirm transcripts appear and AI suggestions stream.
4. Quit; confirm no orphan `python`/`uvicorn` process remains.

## Known sharp edges

- **torch / ctranslate2 / pyannote in a portable runtime** are the main risk. CPU
  wheels are reliable; CUDA needs the matching `TORCH_INDEX_URL` and a GPU box. If
  `pyannote` is troublesome, ship without diarization (it's optional).
- **Unsigned builds** (current plan): Windows SmartScreen → "Windows protected
  your PC" → *More info → Run anyway*. macOS Gatekeeper → right-click → Open, or
  `xattr -dr com.apple.quarantine <app>`. To remove these later, add a Windows
  code-signing cert (`win.certificateFile`) and an Apple Developer ID +
  notarization to the `build` block.
- **First run needs internet** to fetch models. After that it runs offline (use a
  local LLM via the Custom provider for fully offline AI too).
- `scripts/build-server-bundle.mjs` depends on the network and the pinned PBS
  release — bump `PBS_TAG` / `PY_VERSION` deliberately and re-verify asset names.

[python-build-standalone]: https://github.com/astral-sh/python-build-standalone
