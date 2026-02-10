# AsuréFlow

Local, real-time conversation assistant for:
- live transcription (mic + system audio)
- AI response guidance
- claim-level fact checking
- rolling notes
- session history

Built with FastAPI + WebSockets + faster-whisper.

## What It Does
- Captures speech from mic and/or loopback audio.
- Transcribes in real time with optional speaker diarization.
- Generates:
  - `Responses`: practical, fact-grounded reply suggestions.
  - `Fact Check`: supported/contradicted/uncertain claim checks.
  - `Notes`: key points, actions, decisions, risks.
- Saves sessions locally and lets you reload old sessions.
- Supports multiple AI API routes with automatic fallback.

[image of live session view showing transcript + notes/response/fact check side tabs]

## Quick Start (Windows)
1. Install Python `3.11+`.
2. Run `run.bat`.
3. Open Settings, configure at least one API provider/key.

## VS Code Run
1. Open folder in VS Code.
2. Run `F5` with one of:
   - `AI Assistant (no Audio)` (recommended first run)
   - `AI Assistant (with Audio)`

The debug profile runs `.vscode/setup-venv.ps1` first.

## Manual Run
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## API Providers
Primary + fallback providers are supported.

Built-in provider presets:
- `openrouter`
- `openai`
- `canopywave`
- `huggingface`
- `gemini`
- `github_models`
- `custom` (OpenAI-compatible)

Fallback behavior:
- Configure primary API fields in Settings.
- Add additional routes in `Additional API Routes (JSON Array)`.
- Enable `Enable API Fallback`.
- If one route fails/declines, the app tries the next route automatically.

[image of API settings with fallback enabled and additional API routes JSON]

## Environment Variables
- `AI_ASSISTANT_ENABLE_AUDIO` = `0` or `1`
- `AI_ASSISTANT_PORT` = server port (default `8000`)
- `AI_ASSISTANT_USE_WEBVIEW` = `0` or `1`
- `AI_ASSISTANT_CONFIG_PATH` = custom settings file path

Common provider key env vars:
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `CANOPYWAVE_API_KEY`
- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`
- `GEMINI_API_KEY`
- `GITHUB_TOKEN` or `GH_TOKEN`

## Transcription Notes
- Model setting is in UI (`whisper_model_size`).
- Device setting is `cpu` or `cuda`.
- For CUDA on Windows, runtime DLLs must be available (for example `cublas64_12.dll`).
- If CUDA is unavailable at runtime, transcription falls back to CPU.

## Session + Editing
- Sessions are autosaved locally.
- Transcript messages can be edited/deleted from the live chat.
- Notes/responses/fact checks update against current transcript context.

[image of session history panel and load-into-live workflow]

## Project Layout
- `main.py`: server, websocket orchestration, settings, routes
- `backend/`: audio, transcription, diarization, LLM client
- `index.html`, `js/`, `css/`: UI
- `tests/`: unit tests
- `.vscode/`: debug and task helpers

## Development
Run tests:
```powershell
python -m unittest discover -s tests -q
```

Quick syntax checks:
```powershell
python -m py_compile main.py backend\llm.py
node --check js\app.js
```

## Disclaimer
Response/fact-check output is assistive only. Verify important claims independently, especially for legal, medical, or financial decisions.
