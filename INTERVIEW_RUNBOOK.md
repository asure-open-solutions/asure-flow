# Interview Demo Runbook

## Before the call

1. Run `npm run setup` once if this is a fresh checkout.
2. Add one working LLM key to `.env` (OpenRouter is the simplest option).
3. Double-click `start.bat` or run `npm start`.
4. Wait for the client to open. The launcher now waits for server health instead
   of relying on a fixed delay.
5. Confirm the status bar shows the server and LLM as available. Whisper warms
   in the background; the first recording no longer controls API readiness.

## Windows server + Mac client

1. On the Windows PC run `start-server.bat lan`. The window prints the LAN URL.
2. If the Mac cannot reach that URL, allow inbound TCP 8000 in Windows Firewall.
3. On the Mac run `./start-client.sh http://<windows-ip>:8000`.
4. In Settings → Audio, leave Capture Location on **Auto**. Remote clients then
   capture both mic and system audio locally instead of using stale PC devices.
5. Grant macOS Microphone permission and Screen Recording permission when system
   audio is enabled.
6. Confirm the status bar shows network latency, Whisper ready, and an LLM.

## Demo flow

1. Create a new session.
2. Select the Interview or Coding Interview preset.
3. Leave microphone enabled. Enable system audio only when the interviewer's
   audio is playing through this computer.
4. Press **Record**, speak a short test sentence, and confirm it appears.
5. Press `Ctrl+Shift+O` to demonstrate the overlay.
6. Return to the main window, stop recording, then export the session.

## Fast fallbacks

- **Server unavailable:** read the server window's last error, close both
  windows, and run `start.bat` again.
- **LLM unavailable:** open Settings → LLM and verify the provider key and model.
  Transcription continues even without AI insights.
- **No system audio:** disable system audio and continue with the microphone.
  Do not restart a working recording for optional loopback capture.
- **Slow or inaccurate transcription:** choose `tiny` or `small` in
  Settings → Whisper, then restart before the interview.
- **Remote client:** explicitly set `HOST=0.0.0.0` only on a trusted network.
  Local mode intentionally defaults to `127.0.0.1`.
