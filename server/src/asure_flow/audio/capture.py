"""Server-side audio device enumeration and capture via sounddevice."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_sd = None


def _ensure_sounddevice():
    global _sd
    if _sd is None:
        import sounddevice as sd
        _sd = sd
    return _sd


@dataclass
class AudioDevice:
    id: int
    name: str
    channels: int
    sample_rate: float
    is_input: bool
    is_output: bool
    is_loopback: bool = False


def enumerate_devices() -> list[AudioDevice]:
    """List all audio devices available on the server machine.

    Loopback detection per platform:
    - Windows: any output device can be opened via WASAPI loopback.
    - Linux: PulseAudio/PipeWire expose monitor sources as input devices
      with "Monitor of" in the name.
    - macOS: no native loopback; virtual audio devices (BlackHole,
      Soundflower, Loopback) appear as standard input devices.
    """
    sd = _ensure_sounddevice()
    devices = sd.query_devices()
    plat = sys.platform
    result = []

    _mac_virtual = ("blackhole", "soundflower", "loopback", "existential")

    for i, dev in enumerate(devices):
        is_input = dev["max_input_channels"] > 0
        is_output = dev["max_output_channels"] > 0

        if plat == "win32":
            is_loopback = is_output
        elif plat == "linux":
            is_loopback = is_input and "monitor" in dev["name"].lower()
        elif plat == "darwin":
            name_lower = dev["name"].lower()
            is_loopback = is_input and any(v in name_lower for v in _mac_virtual)
        else:
            is_loopback = False

        result.append(AudioDevice(
            id=i,
            name=dev["name"],
            channels=max(dev["max_input_channels"], dev["max_output_channels"]),
            sample_rate=dev["default_samplerate"],
            is_input=is_input,
            is_output=is_output,
            is_loopback=is_loopback,
        ))
    return result


def enumerate_input_devices() -> list[AudioDevice]:
    """List only audio input devices."""
    return [d for d in enumerate_devices() if d.is_input]


def enumerate_output_devices() -> list[AudioDevice]:
    """List audio output devices (available as loopback sources on Windows)."""
    return [d for d in enumerate_devices() if d.is_output]


def enumerate_loopback_devices() -> list[AudioDevice]:
    """List devices capable of loopback/system audio capture on this platform."""
    return [d for d in enumerate_devices() if d.is_loopback]


SAMPLE_RATE = 16000
BLOCK_SIZE = 1024


class ServerAudioCapture:
    """Captures audio from a specific device on the server machine."""

    def __init__(self) -> None:
        self._stream = None
        self._running = False
        self.on_audio: Optional[Callable[[int, bytes], None]] = None

    async def start(self, device_id: int, stream_id: int = 0, loopback: bool = False) -> None:
        sd = _ensure_sounddevice()
        self._running = True
        loop = asyncio.get_event_loop()

        def callback(indata: np.ndarray, frames, time_info, status):
            if status:
                logger.warning("sounddevice status: %s", status)
            # Mono-mix: average all channels
            if indata.shape[1] > 1:
                mono = indata.mean(axis=1)
            else:
                mono = indata[:, 0]
            pcm = (mono * 32767).astype(np.int16).tobytes()
            if self.on_audio:
                loop.call_soon_threadsafe(self.on_audio, stream_id, pcm)

        extra_settings = None
        channels = 1
        if loopback:
            dev_info = sd.query_devices(device_id)
            if sys.platform == "win32":
                # Windows: WASAPI loopback — open the output device with special settings
                try:
                    extra_settings = sd.WasapiSettings(auto_convert=True)
                except AttributeError:
                    logger.warning("WASAPI loopback not available")
                channels = dev_info.get("max_output_channels", 2) or 2
            else:
                # Linux (PulseAudio/PipeWire monitor) / macOS (virtual device):
                # These appear as regular input devices — open as a normal InputStream.
                channels = dev_info.get("max_input_channels", 1) or 1
                logger.info(
                    "Opening loopback device as standard input: %s (channels=%d)",
                    dev_info.get("name", "?"), channels,
                )

        self._stream = sd.InputStream(
            device=device_id,
            samplerate=SAMPLE_RATE,
            channels=channels,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=callback,
            extra_settings=extra_settings,
        )
        self._stream.start()
        logger.info(
            "Server audio capture started: device=%d stream_id=%d loopback=%s",
            device_id, stream_id, loopback,
        )

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Server audio capture stopped")
