import numpy as np
import threading
import queue
import time
import logging
import os
import multiprocessing as mp
from contextlib import suppress
from queue import Full

if os.name != "nt":
    import soundcard as sc
else:
    sc = None

logger = logging.getLogger(__name__)

_SOUNDCARD_NUMPY_COMPAT_APPLIED = False

_SYSTEM_MIX_NAME_HINTS = (
    "stereo mix",
    "what u hear",
    "wave out mix",
    "monitor of",
    "mixage stéréo",
)

def _looks_like_system_mix_device_name(name: str | None) -> bool:
    if not name:
        return False
    n = str(name).casefold()
    return any(pat in n for pat in _SYSTEM_MIX_NAME_HINTS)


def _looks_like_generic_input_device_name(name: str | None) -> bool:
    if not name:
        return False
    n = str(name).casefold()
    return any(
        pat in n
        for pat in (
            "microsoft sound mapper",
            "primary sound capture driver",
        )
    )


def _ensure_soundcard_numpy_compat():
    """
    soundcard<=0.4.x uses numpy.fromstring on a binary buffer on Windows (MediaFoundation).
    NumPy 2.x removed that mode; patch to numpy.frombuffer for compatibility.
    """
    global _SOUNDCARD_NUMPY_COMPAT_APPLIED
    if _SOUNDCARD_NUMPY_COMPAT_APPLIED:
        return

    if sc is None:
        _SOUNDCARD_NUMPY_COMPAT_APPLIED = True
        return

    try:
        major = int(np.__version__.split(".", 1)[0])
    except Exception:
        major = 0

    if major < 2:
        _SOUNDCARD_NUMPY_COMPAT_APPLIED = True
        return

    try:
        import soundcard.mediafoundation as mf  # Windows backend

        if hasattr(mf, "numpy") and hasattr(mf.numpy, "frombuffer"):
            def _fromstring_binary_compat(buffer, dtype=float, count=-1, sep=""):
                if sep not in ("", None):
                    raise ValueError("soundcard compatibility shim only supports binary fromstring()")
                return mf.numpy.frombuffer(buffer, dtype=dtype, count=count).copy()

            mf.numpy.fromstring = _fromstring_binary_compat
            logger.warning(
                "Applied soundcard/NumPy compatibility patch (NumPy>=2): using numpy.frombuffer(...).copy()."
            )
    except Exception as e:
        logger.warning(f"Failed to apply soundcard/NumPy compatibility patch: {e}")
    finally:
        _SOUNDCARD_NUMPY_COMPAT_APPLIED = True


def _com_initialize_for_thread() -> bool:
    if os.name != "nt":
        return False

    try:
        import ctypes

        ole32 = ctypes.OleDLL("ole32")
        COINIT_MULTITHREADED = 0x0
        hr = int(ole32.CoInitializeEx(None, COINIT_MULTITHREADED))
        # S_OK (0) or S_FALSE (1) both require CoUninitialize.
        if hr in (0, 1):
            return True
        # RPC_E_CHANGED_MODE: already initialized with different model; don't uninitialize.
        return False
    except Exception:
        return False


def _com_uninitialize_for_thread():
    if os.name != "nt":
        return

    try:
        import ctypes

        ole32 = ctypes.OleDLL("ole32")
        ole32.CoUninitialize()
    except Exception:
        pass


def _capture_worker(
    device_name: str | None,
    *,
    sample_rate: int,
    block_size: int,
    label: str,
    out_queue,
    stop_event,
):
    """Capture audio in a separate process to isolate native crashes (Windows soundcard can hard-crash)."""
    try:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Keep soundcard out of Windows capture subprocesses entirely; it can hard-crash.
        if os.name != "nt":
            _ensure_soundcard_numpy_compat()

        com_inited = _com_initialize_for_thread()
        try:
            # Windows: use PyAudioWPatch (PortAudio WASAPI + loopback) for both mic + system audio.
            # Do not touch soundcard in this subprocess.
            if os.name == "nt":
                import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]

                with pyaudio.PyAudio() as pa:
                    want_loopback = (label == "loopback")
                    token = (str(device_name).strip() if device_name else "")
                    wasapi_host_api_index = None
                    with suppress(Exception):
                        wasapi_host_api_index = int(pa.get_host_api_info_by_type(pyaudio.paWASAPI).get("index"))

                    dev_info = None
                    if want_loopback:
                        # Prefer explicit loopback device selection (name/index).
                        # If config stored a speaker/output device name (older soundcard path),
                        # try to map it to the corresponding WASAPI loopback device.
                        if token:
                            try:
                                idx = int(token)
                                info = pa.get_device_info_by_index(idx)
                                if info.get("isLoopbackDevice"):
                                    dev_info = info
                                else:
                                    # Map output device -> loopback analogue
                                    dev_info = pa.get_wasapi_loopback_analogue_by_dict(info)
                            except Exception:
                                # Name match against loopback devices
                                for info in pa.get_loopback_device_info_generator():
                                    if str(info.get("name", "")) == token:
                                        dev_info = info
                                        break

                                if dev_info is None:
                                    # Try name match against output devices then map
                                    for i in range(pa.get_device_count()):
                                        info = pa.get_device_info_by_index(i)
                                        if info.get("maxOutputChannels", 0) <= 0:
                                            continue
                                        if str(info.get("name", "")) == token:
                                            try:
                                                dev_info = pa.get_wasapi_loopback_analogue_by_dict(info)
                                            except Exception:
                                                dev_info = None
                                            break

                        if dev_info is None:
                            dev_info = pa.get_default_wasapi_loopback()

                        max_ch = int(dev_info.get("maxInputChannels", 0))
                        channels = 2 if max_ch >= 2 else 1
                    else:
                        # Microphone input
                        def _iter_input_devices():
                            for i in range(pa.get_device_count()):
                                info = pa.get_device_info_by_index(i)
                                if info.get("isLoopbackDevice"):
                                    continue
                                if int(info.get("maxInputChannels", 0) or 0) <= 0:
                                    continue
                                yield info

                        def _prefer_wasapi(infos: list[dict]) -> list[dict]:
                            if wasapi_host_api_index is None:
                                return infos
                            wasapi = [i for i in infos if int(i.get("hostApi", -1)) == wasapi_host_api_index]
                            non = [i for i in infos if int(i.get("hostApi", -1)) != wasapi_host_api_index]
                            return wasapi + non

                        def _find_input_device_by_token(tok: str):
                            if not tok:
                                return None

                            # Allow passing an index (stored as string) for unambiguous selection.
                            try:
                                idx = int(tok)
                                info = pa.get_device_info_by_index(idx)
                                if info.get("isLoopbackDevice"):
                                    return None
                                if int(info.get("maxInputChannels", 0) or 0) <= 0:
                                    return None
                                return info
                            except Exception:
                                pass

                            candidates = list(_iter_input_devices())
                            tok_cf = tok.casefold()

                            # Exact match first.
                            exact = [info for info in candidates if str(info.get("name", "")).casefold() == tok_cf]
                            if exact:
                                if len(exact) == 1:
                                    return exact[0]

                                exact = _prefer_wasapi(exact)

                                # Prefer the current default within matches.
                                with suppress(Exception):
                                    default_info = pa.get_default_input_device_info()
                                    default_idx = int(default_info.get("index"))
                                    for info in exact:
                                        if int(info.get("index")) == default_idx:
                                            return info

                                # Otherwise pick the "best" match.
                                exact.sort(
                                    key=lambda d: (
                                        int(d.get("maxInputChannels", 0) or 0),
                                        float(d.get("defaultSampleRate", 0.0) or 0.0),
                                    ),
                                    reverse=True,
                                )
                                return exact[0]

                            # Fuzzy contains match as a fallback (some Windows drivers include extra suffixes).
                            fuzzy = [info for info in candidates if tok_cf in str(info.get("name", "")).casefold()]
                            if fuzzy:
                                fuzzy = _prefer_wasapi(fuzzy)
                                fuzzy.sort(key=lambda d: int(d.get("maxInputChannels", 0) or 0), reverse=True)
                                return fuzzy[0]

                            return None

                        dev_info = _find_input_device_by_token(token)

                        if dev_info is None:
                            # Prefer WASAPI default input device (PortAudio's generic default can be a mapper/driver).
                            with suppress(Exception):
                                info = pa.get_default_wasapi_device()
                                if (not info.get("isLoopbackDevice")) and int(info.get("maxInputChannels", 0) or 0) > 0:
                                    dev_info = info

                        if dev_info is None:
                            with suppress(Exception):
                                info = pa.get_default_input_device_info()
                                if (not info.get("isLoopbackDevice")) and int(info.get("maxInputChannels", 0) or 0) > 0:
                                    dev_info = info

                        if dev_info is None:
                            candidates = list(_iter_input_devices())
                            candidates = _prefer_wasapi(candidates)
                            dev_info = candidates[0] if candidates else None

                        if dev_info is None:
                            logger.error("Audio [mic]: no usable input devices found")
                            return

                        # Avoid generic "Sound Mapper"/"Primary Sound Capture" devices when auto-selecting.
                        if (not token) and _looks_like_generic_input_device_name(str(dev_info.get("name", ""))):
                            candidates = list(_iter_input_devices())
                            candidates = _prefer_wasapi(candidates)
                            better = None
                            for info in candidates:
                                name_cf = str(info.get("name", "")).casefold()
                                if _looks_like_generic_input_device_name(name_cf):
                                    continue
                                if _looks_like_system_mix_device_name(name_cf):
                                    continue
                                if "microphone" in name_cf or name_cf.startswith("mic"):
                                    better = info
                                    break
                            if better is None:
                                for info in candidates:
                                    name_cf = str(info.get("name", "")).casefold()
                                    if _looks_like_generic_input_device_name(name_cf):
                                        continue
                                    if _looks_like_system_mix_device_name(name_cf):
                                        continue
                                    better = info
                                    break
                            if better is not None:
                                dev_info = better

                        # If the default device is "Stereo Mix"/"What U Hear", prefer an actual microphone.
                        if (not token) and _looks_like_system_mix_device_name(str(dev_info.get("name", ""))):
                            candidates = list(_iter_input_devices())
                            candidates = _prefer_wasapi(candidates)
                            preferred = None
                            for info in candidates:
                                name_cf = str(info.get("name", "")).casefold()
                                if _looks_like_system_mix_device_name(name_cf):
                                    continue
                                if "microphone" in name_cf or name_cf.startswith("mic"):
                                    preferred = info
                                    break
                            if preferred is None:
                                for info in candidates:
                                    name_cf = str(info.get("name", "")).casefold()
                                    if _looks_like_system_mix_device_name(name_cf):
                                        continue
                                    preferred = info
                                    break
                            if preferred is not None:
                                dev_info = preferred

                        max_ch = int(dev_info.get("maxInputChannels", 0) or 0)
                        # Prefer stereo when available: some mics expose signal on only one channel.
                        channels = 2 if max_ch >= 2 else (1 if max_ch >= 1 else 0)

                    if channels <= 0:
                        logger.error(f"Audio [{label}]: selected device has no usable channels")
                        return

                    input_rate = int(round(float(dev_info.get("defaultSampleRate", sample_rate) or sample_rate)))
                    rate_candidates: list[int] = []
                    for r in (input_rate, int(sample_rate), 48000, 44100):
                        try:
                            rr = int(r)
                        except Exception:
                            continue
                        if rr > 0 and rr not in rate_candidates:
                            rate_candidates.append(rr)

                    channel_candidates = [int(channels)]
                    if int(channels) != 1 and int(dev_info.get("maxInputChannels", 0) or 0) >= 1:
                        channel_candidates.append(1)

                    stream = None
                    last_err = None
                    chosen = None
                    for ch in channel_candidates:
                        for rate in rate_candidates:
                            try:
                                frames_per_buffer = int(round(float(block_size) * float(rate) / float(sample_rate)))
                                frames_per_buffer = max(64, frames_per_buffer)
                                stream = pa.open(
                                    format=pyaudio.paInt16,
                                    channels=int(ch),
                                    rate=int(rate),
                                    input=True,
                                    frames_per_buffer=int(frames_per_buffer),
                                    input_device_index=int(dev_info["index"]),
                                )
                                channels = int(ch)
                                input_rate = int(rate)
                                chosen = (channels, input_rate, frames_per_buffer)
                                last_err = None
                                break
                            except Exception as e:
                                last_err = e
                                stream = None
                        if stream is not None:
                            break

                    if stream is None:
                        logger.error(f"Audio [{label}]: failed to open input stream: {last_err}")
                        return
                    channels, input_rate, frames_per_buffer = chosen

                    logger.info(
                        f"Audio [{label}]: using PyAudioWPatch on '{dev_info.get('name', dev_info.get('index'))}' "
                        f"(channels={channels}, in_rate={input_rate}, out_rate={sample_rate}, frames={frames_per_buffer})"
                    )

                    try:
                        while not stop_event.is_set():
                            try:
                                raw = stream.read(int(frames_per_buffer), exception_on_overflow=False)
                            except Exception:
                                logger.exception(f"Audio [{label}]: stream.read() failed")
                                break

                            data = np.frombuffer(raw, dtype=np.int16)
                            if channels > 1:
                                frames = data.reshape(-1, int(channels))
                                if label == "mic":
                                    # Some drivers report stereo but only one channel is usable (or channels differ);
                                    # pick the loudest channel to avoid accidental cancellation/near-silence.
                                    f32 = frames.astype(np.float32)
                                    rms = np.sqrt(np.mean(f32 * f32, axis=0))
                                    ch_idx = int(np.argmax(rms)) if rms.size else 0
                                    data = frames[:, ch_idx]
                                else:
                                    data = frames.mean(axis=1)
                            data = data.astype(np.float32) / 32768.0

                            # Resample to Whisper target sample rate if needed.
                            if input_rate != int(sample_rate):
                                target_len = int(block_size)
                                if target_len > 0 and data.size > 0:
                                    if input_rate % int(sample_rate) == 0:
                                        dec = input_rate // int(sample_rate)
                                        data = data[::dec]
                                    else:
                                        x_old = np.linspace(0.0, 1.0, num=int(data.size), endpoint=False)
                                        x_new = np.linspace(0.0, 1.0, num=int(target_len), endpoint=False)
                                        data = np.interp(x_new, x_old, data).astype(np.float32)
                                    if data.size > target_len:
                                        data = data[:target_len]
                                    elif data.size < target_len:
                                        data = np.pad(data, (0, target_len - data.size))

                            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                            data = np.clip(data, -1.0, 1.0)
                            try:
                                out_queue.put(data, block=False)
                            except Full:
                                # Keep newest audio: drop one oldest chunk and retry.
                                with suppress(Exception):
                                    out_queue.get_nowait()
                                with suppress(Exception):
                                    out_queue.put(data, block=False)
                    finally:
                        with suppress(Exception):
                            stream.stop_stream()
                        with suppress(Exception):
                            stream.close()
                return

            if sc is None:
                return

            mic = None
            if device_name:
                for d in sc.all_microphones(include_loopback=True):
                    if d.name == device_name:
                        mic = d
                        break
            if mic is None:
                mic = sc.default_microphone()

            if mic is None:
                return

            with mic.recorder(samplerate=sample_rate, channels=1) as rec:
                while not stop_event.is_set():
                    data = rec.record(numframes=block_size)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    data = np.clip(data, -1.0, 1.0)
                    payload = data.flatten()
                    try:
                        out_queue.put(payload, block=False)
                    except Full:
                        with suppress(Exception):
                            out_queue.get_nowait()
                        with suppress(Exception):
                            out_queue.put(payload, block=False)
        finally:
            if com_inited:
                _com_uninitialize_for_thread()
    except Exception:
        # Best-effort: parent process will notice process exit.
        logger.exception(f"Audio [{label}]: capture worker crashed")
        return


class AudioManager:
    def __init__(self, sample_rate=16000, block_size=2048):
        self.sample_rate = sample_rate
        self.block_size = block_size
        # Keep enough buffered audio to survive Whisper inference bursts without dropping.
        self.max_local_queue_chunks = 320
        self.max_mp_queue_chunks = 1200
        self.queue_batch_chunks = 6
        self.queue_backlog_warn_chunks = 96
        self.queue_backlog_warn_interval_s = 2.0
        self._last_backlog_warn_ts = {"mic": 0.0, "loopback": 0.0}

        # Use multiprocessing for capture to isolate native crashes.
        self._mp_ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
        self._mic_proc = None
        self._loop_proc = None
        self._mic_stop = None
        self._loop_stop = None
        self._mic_q_mp = None
        self._loop_q_mp = None

        # Local queues consumed by transcribers.
        self.mic_queue = queue.Queue(maxsize=self.max_local_queue_chunks)
        self.loopback_queue = queue.Queue(maxsize=self.max_local_queue_chunks)

        # Pump threads (main process) to forward mp queue -> local queue.
        self._pump_threads = []

        self.is_recording = False
        self.mic_active = False
        self.loopback_active = False

    def list_microphones(self):
        """Returns a list of available microphones."""
        if os.name == "nt":
            try:
                import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]

                devices = []
                with pyaudio.PyAudio() as pa:
                    for i in range(pa.get_device_count()):
                        info = pa.get_device_info_by_index(i)
                        name = str(info.get("name", ""))
                        if not name:
                            continue
                        is_loop = bool(info.get("isLoopbackDevice", False))
                        is_system_mix = (not is_loop) and _looks_like_system_mix_device_name(name)
                        max_in = int(info.get("maxInputChannels", 0) or 0)
                        if (not is_loop) and max_in <= 0:
                            continue
                        devices.append(
                            {
                                # Use the device index as a stable identifier on Windows.
                                # Device names can be duplicated and are not stable across driver changes.
                                "id": str(info.get("index", i)),
                                "name": name,
                                "is_loopback": is_loop,
                                "is_system_mix": is_system_mix,
                                "max_input_channels": max_in,
                                "max_output_channels": int(info.get("maxOutputChannels", 0) or 0),
                            }
                        )
                return devices
            except Exception as e:
                logger.error(f"Failed to list audio devices via PyAudioWPatch: {e}")
                return []

        if sc is None:
            return []

        return sc.all_microphones(include_loopback=True)

    def get_device_by_id(self, device_id):
        """
        Helper to find a device by its ID (name). 
        In soundcard, devices are identified by object, so we match name.
        """
        if os.name == "nt":
            # Not used for Windows capture (handled via PyAudioWPatch in the subprocess).
            return None

        if sc is None:
            return None

        mics = sc.all_microphones(include_loopback=True)
        for mic in mics:
            if mic.name == device_id:
                return mic
        return None

    def start_recording(self, mic_name=None, loopback_name=None, start_mic=True, start_loopback=False):
        if os.name != "nt":
            _ensure_soundcard_numpy_compat()
        self.stop_recording()

        # Mark recording active early so pumps run.
        self.is_recording = True
        self.mic_active = False
        self.loopback_active = False
        started_any = False

        # Create mp primitives per run.
        self._mic_stop = self._mp_ctx.Event()
        self._loop_stop = self._mp_ctx.Event()
        self._mic_q_mp = self._mp_ctx.Queue(maxsize=self.max_mp_queue_chunks)
        self._loop_q_mp = self._mp_ctx.Queue(maxsize=self.max_mp_queue_chunks)
        
        if start_mic:
            if os.name == "nt":
                logger.info(f"Starting mic recording (PyAudioWPatch) (subprocess)")
                self._mic_proc = self._mp_ctx.Process(
                    target=_capture_worker,
                    kwargs={
                        "device_name": mic_name,
                        "sample_rate": int(self.sample_rate),
                        "block_size": int(self.block_size),
                        "label": "mic",
                        "out_queue": self._mic_q_mp,
                        "stop_event": self._mic_stop,
                    },
                    daemon=True,
                )
                self._mic_proc.start()
                started_any = True
            else:
                if sc is None:
                    logger.warning("No soundcard backend available for mic capture on this platform.")
                else:
                    mic = self.get_device_by_id(mic_name) if mic_name else sc.default_microphone()
                    if mic:
                        logger.info(f"Starting mic recording on: {mic.name} (subprocess)")
                        self._mic_proc = self._mp_ctx.Process(
                            target=_capture_worker,
                            kwargs={
                                "device_name": mic.name,
                                "sample_rate": int(self.sample_rate),
                                "block_size": int(self.block_size),
                                "label": "mic",
                                "out_queue": self._mic_q_mp,
                                "stop_event": self._mic_stop,
                            },
                            daemon=True,
                        )
                        self._mic_proc.start()
                        started_any = True
                    else:
                        logger.warning("No microphone found or selected.")

        if start_loopback:
            if os.name == "nt":
                logger.info("Starting loopback recording (PyAudioWPatch) (subprocess)")
                self._loop_proc = self._mp_ctx.Process(
                    target=_capture_worker,
                    kwargs={
                        "device_name": loopback_name,
                        "sample_rate": int(self.sample_rate),
                        "block_size": int(self.block_size),
                        "label": "loopback",
                        "out_queue": self._loop_q_mp,
                        "stop_event": self._loop_stop,
                    },
                    daemon=True,
                )
                self._loop_proc.start()
                started_any = True
            else:
                loopback = None
                if loopback_name:
                    loopback = self.get_device_by_id(loopback_name)
                else:
                    try:
                        mics = sc.all_microphones(include_loopback=True)
                        for m in mics:
                            if getattr(m, "isloopback", False):
                                loopback = m
                                break
                        if not loopback:
                            for m in mics:
                                if "loopback" in m.name.lower():
                                    loopback = m
                                    break
                    except Exception as e:
                        logger.warning(f"Error finding loopback: {e}")

                if loopback:
                    logger.info(f"Starting loopback recording on: {loopback.name} (subprocess)")
                    self._loop_proc = self._mp_ctx.Process(
                        target=_capture_worker,
                        kwargs={
                            "device_name": loopback.name,
                            "sample_rate": int(self.sample_rate),
                            "block_size": int(self.block_size),
                            "label": "loopback",
                            "out_queue": self._loop_q_mp,
                            "stop_event": self._loop_stop,
                        },
                        daemon=True,
                    )
                    self._loop_proc.start()
                    started_any = True
                else:
                    logger.warning("Could not identify a loopback microphone device; skipping loopback recording.")

        # Start pump threads to forward mp queue -> local queue.
        def _pump(mp_q, local_q, proc_getter, label: str, first_chunk_event: threading.Event | None):
            pumped = 0
            dropped = 0
            while self.is_recording:
                proc = proc_getter()
                if proc is not None and (not proc.is_alive()) and mp_q.empty():
                    logger.error(f"Audio [{label}]: capture process exited.")
                    break
                try:
                    item = mp_q.get(timeout=0.1)
                    try:
                        local_q.put(item, block=False)
                    except queue.Full:
                        # Queue protection: if full, drop only one oldest block and keep newest.
                        with suppress(Exception):
                            local_q.get_nowait()
                        try:
                            local_q.put(item, block=False)
                        except queue.Full:
                            dropped += 1
                        else:
                            dropped += 1
                        if dropped % 100 == 0:
                            approx_lag_s = (float(local_q.qsize()) * float(self.block_size)) / float(max(1, self.sample_rate))
                            logger.warning(
                                f"Audio [{label}]: dropping stale chunks to keep up (dropped={dropped}, "
                                f"qsize={local_q.qsize()}, lag~{approx_lag_s:.1f}s)."
                            )
                    pumped += 1
                    if pumped == 1:
                        logger.info(f"Audio [{label}]: first chunk received from subprocess")
                        if first_chunk_event is not None:
                            first_chunk_event.set()
                except Exception:
                    continue

        mic_first = threading.Event() if self._mic_proc is not None else None
        loop_first = threading.Event() if self._loop_proc is not None else None

        if self._mic_proc is not None:
            t = threading.Thread(
                target=_pump,
                args=(self._mic_q_mp, self.mic_queue, lambda: self._mic_proc, "mic", mic_first),
                daemon=True,
            )
            t.start()
            self._pump_threads.append(t)

        if self._loop_proc is not None:
            t = threading.Thread(
                target=_pump,
                args=(self._loop_q_mp, self.loopback_queue, lambda: self._loop_proc, "loopback", loop_first),
                daemon=True,
            )
            t.start()
            self._pump_threads.append(t)

        # Best-effort health check: ensure each started stream produces at least one chunk quickly.
        # This prevents the UI from thinking audio is running when the subprocess immediately died.
        deadline = time.time() + 1.5
        if self._mic_proc is not None and mic_first is not None:
            remaining = max(0.0, deadline - time.time())
            mic_first.wait(timeout=remaining)
            if (not mic_first.is_set()) and (self._mic_proc is not None) and (not self._mic_proc.is_alive()):
                logger.error("Audio [mic]: failed to start (no audio received)")
                with suppress(Exception):
                    self._mic_stop.set()
                with suppress(Exception):
                    self._mic_proc.join(timeout=0.5)
                self._mic_proc = None

        if self._loop_proc is not None and loop_first is not None:
            remaining = max(0.0, deadline - time.time())
            loop_first.wait(timeout=remaining)
            if (not loop_first.is_set()) and (self._loop_proc is not None) and (not self._loop_proc.is_alive()):
                logger.error("Audio [loopback]: failed to start (no audio received)")
                with suppress(Exception):
                    self._loop_stop.set()
                with suppress(Exception):
                    self._loop_proc.join(timeout=0.5)
                self._loop_proc = None

        self.mic_active = (self._mic_proc is not None) and bool(self._mic_proc.is_alive())
        self.loopback_active = (self._loop_proc is not None) and bool(self._loop_proc.is_alive())
        self.is_recording = self.mic_active or self.loopback_active
        if not started_any:
            logger.warning("Audio recording did not start (no streams opened).")
        return {
            "mic": {"requested": bool(start_mic), "active": bool(self.mic_active), "first_chunk": bool(mic_first and mic_first.is_set())},
            "loopback": {"requested": bool(start_loopback), "active": bool(self.loopback_active), "first_chunk": bool(loop_first and loop_first.is_set())},
        }

    def stop_recording(self):
        self.is_recording = False
        self.mic_active = False
        self.loopback_active = False

        # Signal subprocesses.
        try:
            if self._mic_stop is not None:
                self._mic_stop.set()
            if self._loop_stop is not None:
                self._loop_stop.set()
        except Exception:
            pass

        # Join pump threads.
        for t in self._pump_threads:
            t.join(timeout=1.0)
        self._pump_threads = []

        # Join subprocesses.
        for p in (self._mic_proc, self._loop_proc):
            if p is not None:
                p.join(timeout=1.5)

        self._mic_proc = None
        self._loop_proc = None
        self._mic_stop = None
        self._loop_stop = None
        self._mic_q_mp = None
        self._loop_q_mp = None

        # Clear any buffered audio so the next session starts fresh.
        try:
            while not self.mic_queue.empty():
                self.mic_queue.get_nowait()
        except Exception:
            pass
        try:
            while not self.loopback_queue.empty():
                self.loopback_queue.get_nowait()
        except Exception:
            pass

    # NOTE: capture happens in subprocess via _capture_worker.
            
    def _warn_if_backlogged(self, q: queue.Queue, label: str) -> None:
        try:
            size = int(q.qsize())
        except Exception:
            return
        if size < int(self.queue_backlog_warn_chunks):
            return

        now_ts = time.time()
        last = float(self._last_backlog_warn_ts.get(label, 0.0) or 0.0)
        if (now_ts - last) < float(self.queue_backlog_warn_interval_s):
            return
        self._last_backlog_warn_ts[label] = now_ts
        approx_lag_s = (float(size) * float(self.block_size)) / float(max(1, self.sample_rate))
        logger.warning(f"Audio [{label}]: backlog high (qsize={size}, lag~{approx_lag_s:.1f}s)")

    def _coalesce_queue_chunk(self, q: queue.Queue, first_item) -> np.ndarray:
        base = np.asarray(first_item, dtype=np.float32).reshape(-1)
        if base.size == 0:
            return base

        parts = [base]
        for _ in range(max(0, int(self.queue_batch_chunks) - 1)):
            try:
                nxt = q.get_nowait()
            except queue.Empty:
                break
            arr = np.asarray(nxt, dtype=np.float32).reshape(-1)
            if arr.size:
                parts.append(arr)

        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def get_mic_data(self):
        """Yields audio data from mic queue."""
        while self.is_recording or not self.mic_queue.empty():
            if self._mic_proc is not None and (not self._mic_proc.is_alive()) and self.mic_queue.empty():
                break
            self._warn_if_backlogged(self.mic_queue, "mic")
            try:
                first = self.mic_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            yield self._coalesce_queue_chunk(self.mic_queue, first)

    def get_loopback_data(self):
        """Yields audio data from loopback queue."""
        while self.is_recording or not self.loopback_queue.empty():
            if self._loop_proc is not None and (not self._loop_proc.is_alive()) and self.loopback_queue.empty():
                break
            self._warn_if_backlogged(self.loopback_queue, "loopback")
            try:
                first = self.loopback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            yield self._coalesce_queue_chunk(self.loopback_queue, first)
