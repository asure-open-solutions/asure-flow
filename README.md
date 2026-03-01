<div align="center">

<!-- [image of logo — AsuréFlow logo, clean SVG, ~120px height] -->

# AsuréFlow

**Your AI copilot for every conversation.**

Real-time transcription, agentic AI analysis, and an invisible overlay —<br/>
all running locally on your machine.

<!-- [image of hero-screenshot — main app window showing live transcript on the left and AI insights panel on the right, dark theme] -->

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)]()
[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)]()
[![Electron](https://img.shields.io/badge/electron-34-47848F?logo=electron&logoColor=white)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)]()

[Download](#quick-start) · [Quick Start](#quick-start) · [How It Works](#how-it-works) · [Contributing](#contributing)

</div>

---

> Most transcription tools record your meetings and email you a summary **afterwards**.
> AsuréFlow works **during** the conversation — fact-checking claims, suggesting responses, extracting notes, and searching the web — all in real time, through an overlay that's **invisible to screen share**.

---

## Why AsuréFlow

<!-- [image of feature-pillars — a 2x2 grid showing: (1) AI agent activity panel with tool calls, (2) overlay floating over a Zoom call, (3) local transcription with no cloud icon, (4) the app running alongside Discord/phone/podcast] -->

<table>
<tr>
<td width="25%" align="center">
<h3>🧠 Real-Time Agentic AI</h3>
<p>Not just transcription. An AI agent with 8 tools — fact-checking, web search, code analysis, deep thinking — reasoning in real time as the conversation happens.</p>
</td>
<td width="25%" align="center">
<h3>👁️‍🗨️ Invisible Overlay</h3>
<p>An always-on-top HUD that is completely invisible to screen capture and recording software. Your AI assistant that nobody else can see.</p>
</td>
<td width="25%" align="center">
<h3>🔒 100% Local Transcription</h3>
<p>Audio never leaves your machine. faster-whisper runs on-device with GPU acceleration. Optional PII redaction. No cloud dependency.</p>
</td>
<td width="25%" align="center">
<h3>🌐 Works Everywhere</h3>
<p>Not limited to Zoom or Teams. Works with any audio source — phone calls, in-person meetings, Discord, podcasts, lectures, interviews.</p>
</td>
</tr>
</table>

---

## How AsuréFlow Compares

| Feature | AsuréFlow | Otter.ai | Fireflies.ai | Fathom | Granola | Read.ai | Krisp |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Real-time AI assistance** | ✅ | After meeting | After meeting | After meeting | After meeting | After meeting | ❌ |
| **Agentic AI (tools)** | ✅ 8 tools | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Local transcription** | ✅ | ❌ Cloud | ❌ Cloud | ❌ Cloud | ❌ Cloud | ❌ Cloud | ✅ |
| **Invisible overlay** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Screen-share safe** | ✅ | N/A | N/A | N/A | N/A | N/A | N/A |
| **Works with any app** | ✅ | Zoom/Meet/Teams | Zoom/Meet/Teams | Zoom/Meet/Teams | Zoom/Meet/Teams | Zoom/Meet/Teams | VoIP only |
| **Speaker diarization** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Real-time fact-checking** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Web search in context** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Code analysis** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Multi-provider LLM** | ✅ 6 providers | Proprietary | Proprietary | GPT only | GPT only | Proprietary | N/A |
| **Semantic search** | ✅ | ✅ | ✅ | Partial | ❌ | ✅ | ❌ |
| **PII redaction** | ✅ | Partial | Partial | ❌ | ❌ | Partial | ❌ |
| **Open source** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Pricing** | **Free** | $16.99/mo+ | $19/mo+ | Free (limited) | $12/mo+ | $19.75/mo+ | $8/mo+ |

---

## Features

### Agentic AI That Actually Does Things

AsuréFlow doesn't just summarize — it reasons. The AI agent has access to tools it can chain together across multiple iterations:

| Tool | What It Does |
|:---|:---|
| **Fact Check** | Verifies claims in real time with verdicts: supported, contradicted, or uncertain. Detects logical fallacies. |
| **Suggest Response** | Generates context-aware replies you can actually say. Tone control: professional, casual, assertive, diplomatic, empathetic. |
| **Extract Notes** | Pulls action items (with owners + due dates), decisions, key facts, and risks into structured notes. |
| **Web Search** | Searches the web via DuckDuckGo with source credibility scoring to verify facts live. |
| **Search Transcript** | Semantic search through the current conversation powered by sentence-transformers. |
| **Search Sessions** | Searches across all past sessions — find previously discussed topics, people, and decisions. |
| **Code Analysis** | Detects code in conversations, formats it, identifies language, and analyzes correctness + Big-O complexity. |
| **Deep Think** | Step-by-step reasoning for complex or nuanced topics before responding. |

<!-- [image of agent-activity — the agent activity panel showing a chain of tool calls: fact_check → web_search → suggest_response, with structured results for each] -->

### Situation-Specific AI Presets

Switch AI behavior instantly based on what you're doing:

| Preset | Optimized For |
|:---|:---|
| **General** | Balanced assistant for any conversation. |
| **Meeting** | Action items, decisions, meeting notes. Tracks who owns what. |
| **Interview** | Response suggestions using the STAR method. Confidence coaching. |
| **Debate** | Aggressive fact-checking, logical fallacy detection, evidence-based counterarguments. |
| **Coding Interview** | Code analysis, Big-O complexity, edge cases, algorithm suggestions. |

Or write a **custom system prompt** for your specific use case.

### Screen-Share Safe Overlay

Toggle an always-on-top overlay with `Ctrl+Shift+O` that floats above any application. On Windows, it uses `WDA_EXCLUDEFROMCAPTURE` to be **completely invisible** in screen recordings and screen shares — not blacked out, not blurred, *invisible*.

| Platform | Behavior |
|:---|:---|
| **Windows 10 2004+** | Fully invisible in all screen captures |
| **macOS** | Appears as black rectangle in captures |
| **Linux** | Electron content protection (compositor-dependent) |

Two overlay modes: **Top Bar** (compact horizontal strip) and **Cards** (draggable floating insight cards you can position anywhere).

<!-- [image of overlay-mode — the overlay in "cards" mode floating over a video call or IDE, showing a live transcript card and a suggestion card] -->

### Speaker Diarization

Powered by pyannote.audio 3.1, AsuréFlow identifies and labels different speakers in the conversation. Click to rename speakers to real names — labels persist across the entire session.

<!-- [image of diarization — the transcript panel showing a conversation between multiple speakers, each color-coded with names like "Sarah (Engineering Lead)" and "Mike (PM)"] -->

### Semantic Search Across All Conversations

Find anything ever discussed. Semantic search powered by sentence-transformers finds relevant content even when exact words don't match. Search within a session or across your entire conversation history. Entity-aware — finds people, projects, and decisions by name.

### Your Models, Your Choice

Configure up to 6 LLM providers with automatic failover. If one goes down, AsuréFlow seamlessly switches to the next. Drag to reorder priority.

**Supported providers:** OpenRouter · OpenAI · Google Gemini · HuggingFace · GitHub Models · Custom endpoint (Ollama, LM Studio, or any OpenAI-compatible API)

### Privacy First

| Concern | How It's Handled |
|:---|:---|
| **Audio data** | Processed locally by faster-whisper. Never uploaded. |
| **LLM communication** | Transcript text (not audio) is sent to your chosen LLM provider for AI features. |
| **Privacy mode** | One toggle disables web search + enables PII redaction. |
| **PII redaction** | Detects and redacts emails, phone numbers, SSNs, and credit card numbers. |
| **Screen capture** | Content protection makes the overlay invisible to screen recording. |
| **Data storage** | Sessions stored locally as JSON. No cloud account, no telemetry. |
| **API keys** | Stored in local config. Masked in the UI. Never logged. |

> **Tip:** For maximum privacy, use a local LLM via the Custom endpoint (e.g., Ollama or LM Studio) — then nothing leaves your machine at all.

### Export & Follow-Up

- Export sessions as **JSON** or **Markdown**
- Generate AI-powered follow-up drafts: **email**, **Slack/Teams message**, or **structured summary** — with action items, decisions, and next steps pre-populated from your conversation notes

---

## How It Works

<!-- [image of architecture-diagram — a horizontal flow diagram showing: Mic+System Audio → WebSocket → faster-whisper (with VAD) → Transcript → Agentic AI Loop (with tool icons) → Overlay/UI] -->

1. **Capture** — Microphone and system audio are captured simultaneously as 16kHz mono PCM via AudioWorklet or system loopback.

2. **Stream** — Audio chunks stream over WebSocket with a 1-byte stream ID prefix (`0x00` = mic, `0x01` = system).

3. **Transcribe** — The server accumulates audio in a VAD-aware buffer (Silero VAD). When a natural pause is detected, the buffer flushes to faster-whisper. No fixed intervals — no cutting sentences mid-word.

4. **Diarize** *(optional)* — Audio passes through pyannote.audio's speaker diarization pipeline to identify who is speaking.

5. **Analyze** — Each transcript segment enters the agentic AI loop, which streams through your configured LLM with access to 8 specialized tools. The agent can chain multiple tools across up to 5 iterations per segment.

6. **Display** — Results stream back in real time and render in the transcript panel, insights drawer, or floating overlay.

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.13 recommended)
- **Node.js 18+**
- **At least one LLM API key** ([OpenRouter](https://openrouter.ai) is the easiest — one key, many models)
- *(Optional)* NVIDIA GPU with CUDA for faster transcription and diarization

### Install

```bash
git clone https://github.com/asure-solutions/asure-flow.git
cd asure-flow
./scripts/setup.sh
```

### Configure

Create a `.env` file in the project root (or configure everything through the Settings UI after launching):

```env
# At least one LLM provider API key:
OPENROUTER_API_KEY=sk-or-...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=AI...

# Whisper (defaults work well for most setups):
# WHISPER_MODEL=large-v3-turbo
# WHISPER_DEVICE=cuda
```

### Run

```bash
./scripts/dev.sh
```

The server starts at `http://localhost:8000` and the Electron app opens automatically.

<!-- [image of first-launch — the app on first launch showing an empty session with a "Start Recording" button and the settings panel open for initial LLM configuration] -->

---

## Configuration

### LLM Providers

| Provider | Env Variable | Default Model |
|:---|:---|:---|
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4.1` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| HuggingFace | `HF_API_KEY` | `meta-llama/Llama-3.1-70B-Instruct` |
| GitHub Models | `GITHUB_TOKEN` | `gpt-4o` |
| Custom | `CUSTOM_API_BASE` + `CUSTOM_MODEL` | *(your choice)* |

Providers are tried in priority order with automatic failover. Configure order and models in the Settings UI or via `.env`.

<!-- [image of settings — the settings panel showing the LLM provider configuration with drag-to-reorder priority list and API key fields] -->

### Whisper Models

| Model | Speed | Accuracy | VRAM |
|:---|:---|:---|:---|
| `tiny` | Fastest | Lower | ~1 GB |
| `base` | Fast | Good | ~1 GB |
| `small` | Moderate | Better | ~2 GB |
| `medium` | Slower | High | ~5 GB |
| `large-v3-turbo` *(default)* | Fast | Highest | ~6 GB |

### Optional Features

| Feature | Requirement | Description |
|:---|:---|:---|
| Speaker diarization | HuggingFace token + `pyannote.audio` | Identifies who is speaking |
| Semantic search | `sentence-transformers` | Embedding-based search across sessions |
| Server audio capture | `sounddevice` | Server-side loopback capture (alternative to client capture) |

### Keyboard Shortcuts

| Shortcut | Action |
|:---|:---|
| `Ctrl+Shift+O` | Toggle overlay mode |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+I` | Toggle insights drawer |

---

## Architecture

```
asure-flow/
├── server/                Python 3.13 · FastAPI · faster-whisper · LiteLLM
│   └── src/asure_flow/
│       ├── agent/         Agentic AI loop, tools, presets, context management
│       ├── api/           REST endpoints (sessions, config, export)
│       ├── audio/         Server-side audio capture (sounddevice)
│       ├── memory/        Entity + topic extraction
│       ├── safety/        PII detection and redaction
│       ├── search/        Semantic search (sentence-transformers + Faiss)
│       ├── sessions/      Session CRUD, models, JSON storage
│       ├── transcription/ faster-whisper engine, VAD buffer, diarization
│       └── ws/            WebSocket handlers (audio streaming, session events)
│
├── client/                Electron 34 · React 19 · Vite · Tailwind CSS 4
│   ├── electron/          Main process, overlay window, content protection, tray
│   └── src/
│       ├── components/    UI (transcript, notes, settings, overlay modes)
│       ├── services/      Audio capture, WebSocket clients, REST client
│       └── stores/        Zustand state management
│
└── scripts/               Setup and dev scripts
```

**Communication:**
- `/ws/audio` — Binary WebSocket for PCM audio streaming (1-byte stream ID prefix)
- `/ws/session/{id}` — JSON WebSocket for AI events, transcription, and config sync
- `/api/*` — REST for sessions CRUD, configuration, search, export, and audio devices

---

## Contributing

Contributions are welcome!

```bash
# Clone and install
git clone https://github.com/asure-solutions/asure-flow.git
cd asure-flow
./scripts/setup.sh

# Development mode (hot reload for server + client)
./scripts/dev.sh
```

**Where to start:**
- Add new AI tools → `server/src/asure_flow/agent/tools.py`
- UI components → `client/src/components/`
- Overlay behavior → `client/electron/overlay.ts`

---

## License

[MIT](LICENSE)

---

<div align="center">

Built by **Asuré Solutions**

<!-- [image of footer-screenshot — a polished, wide shot of the overlay mode in action over a real application, showing the transcript and a suggestion card] -->

If AsuréFlow helps you, consider giving it a ⭐

</div>
