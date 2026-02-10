from openai import AsyncOpenAI
import logging
import json

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self,
        api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-3.5-turbo",
        default_headers=None,
        *,
        fallback_routes=None,
        failover_enabled: bool = True,
    ):
        self.failover_enabled = bool(failover_enabled)
        self._active_endpoint_index = 0
        self.endpoints = []
        self._config_signature = None

        self._add_endpoint(
            {
                "provider": "primary",
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
                "api_extra_headers": default_headers,
            }
        )
        for route in (fallback_routes or []):
            self._add_endpoint(route)

        if not self.endpoints:
            raise ValueError("LLMClient requires at least one endpoint with an API key.")

        self._refresh_compat_fields()
        self.history = []
        self.system_prompt = "You are a helpful AI assistant running on the user's desktop. Keep your answers concise and helpful."

    def _endpoint_signature(self) -> tuple:
        sig = []
        for ep in self.endpoints:
            headers = ep.get("api_extra_headers") or {}
            try:
                hdr_json = json.dumps(headers, sort_keys=True, separators=(",", ":"))
            except Exception:
                hdr_json = "{}"
            sig.append(
                (
                    str(ep.get("provider") or ""),
                    str(ep.get("base_url") or ""),
                    str(ep.get("model") or ""),
                    str(ep.get("api_key") or ""),
                    hdr_json,
                )
            )
        return tuple(sig)

    def set_config_signature(self, sig) -> None:
        self._config_signature = sig

    def get_config_signature(self):
        return self._config_signature

    def _add_endpoint(self, route: dict) -> None:
        if not isinstance(route, dict):
            return
        api_key = str(route.get("api_key") or "").strip()
        if not api_key:
            return

        base_url = str(route.get("base_url") or "").strip()
        model = str(route.get("model") or "").strip()
        if not base_url or not model:
            return

        headers = route.get("api_extra_headers", route.get("default_headers"))
        if not isinstance(headers, dict):
            headers = {}
        provider = str(route.get("provider") or "custom").strip().lower() or "custom"

        ep = {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "api_extra_headers": headers,
            "client": AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=headers,
            ),
        }

        # Keep first occurrence by unique connection tuple.
        for cur in self.endpoints:
            if (
                cur.get("base_url") == ep["base_url"]
                and cur.get("model") == ep["model"]
                and cur.get("api_key") == ep["api_key"]
                and cur.get("api_extra_headers") == ep["api_extra_headers"]
            ):
                return

        self.endpoints.append(ep)

    def _refresh_compat_fields(self) -> None:
        idx = self._active_endpoint_index
        if idx < 0 or idx >= len(self.endpoints):
            idx = 0
            self._active_endpoint_index = 0
        ep = self.endpoints[idx]
        self.api_key = ep["api_key"]
        self.base_url = ep["base_url"]
        self.default_headers = ep["api_extra_headers"]
        self.model = ep["model"]
        self.client = ep["client"]

    async def chat_create(self, **kwargs):
        if not self.endpoints:
            raise RuntimeError("No LLM endpoints configured")

        errors = []
        attempt_count = len(self.endpoints) if self.failover_enabled else 1
        for idx in range(attempt_count):
            ep = self.endpoints[idx]
            req = dict(kwargs)
            req["model"] = ep["model"]
            try:
                resp = await ep["client"].chat.completions.create(**req)
                prev_idx = self._active_endpoint_index
                self._active_endpoint_index = idx
                self._refresh_compat_fields()
                if idx != prev_idx:
                    logger.warning(
                        "LLM failover selected endpoint #%s (%s %s)",
                        idx + 1,
                        ep["provider"],
                        ep["base_url"],
                    )
                return resp
            except Exception as e:
                errors.append(
                    f"#{idx + 1} {ep['provider']} {ep['base_url']} ({ep['model']}): {e}"
                )
                if idx + 1 < attempt_count:
                    logger.warning(
                        "LLM endpoint failed, trying fallback #%s: %s (%s) -> %s",
                        idx + 2,
                        ep["provider"],
                        ep["base_url"],
                        e,
                    )
                continue

        if errors:
            raise RuntimeError("All configured LLM APIs failed. " + " | ".join(errors))
        raise RuntimeError("LLM request failed")

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    @staticmethod
    def _normalize_entity(entity: str | None) -> str:
        e = (entity or "").strip().lower()
        if e in ("assistant", "ai"):
            return "assistant"
        if e in ("third_party", "third-party", "loopback") or e.startswith("third_party"):
            return "third_party"
        if e in ("user", "you"):
            return "you"
        return e or "you"

    @staticmethod
    def _label_from_message(message: dict) -> str:
        role = (message.get("role") or "").strip().lower()
        if role == "assistant":
            return "AI"
        name_raw = message.get("name") or ""
        name = name_raw.strip() if isinstance(name_raw, str) else str(name_raw).strip()
        name_low = name.lower()
        if name_low.startswith("third_party_") and name_low != "third_party":
            # Name is sanitized (OpenAI constraint). Support either:
            # - third_party_spk1 -> "Speaker 1"
            # - third_party_Alice__spk1 -> "Alice"
            suffix = name[len("third_party_") :]
            if "__" in suffix:
                label_part = suffix.split("__", 1)[0]
                label = label_part.replace("_", " ").strip()
                if label:
                    return label
            if suffix.lower().startswith("spk"):
                digits = "".join(ch for ch in suffix if ch.isdigit())
                if digits:
                    return f"Speaker {digits}"
        if name_low.startswith("third_party_spk"):
            digits = "".join(ch for ch in name_low if ch.isdigit())
            if digits:
                return f"Speaker {digits}"
            return "Third-Party"
        if name_low == "third_party":
            return "Third-Party"
        if name_low == "you":
            return "You"
        return "User"

    @staticmethod
    def _safe_openai_name_component(text: str, *, max_len: int = 40) -> str:
        # OpenAI name must be limited to [a-zA-Z0-9_-]. Use underscore for spaces.
        t = (text or "").strip().replace(" ", "_")
        t = "".join(ch for ch in t if ch.isalnum() or ch in ("_", "-"))
        while "__" in t:
            t = t.replace("__", "_")
        t = t.strip("_-")
        if max_len:
            t = t[: int(max_len)]
        return t

    def add_transcript_message(self, entity: str, content: str, *, speaker_id: str | None = None, speaker_label: str | None = None):
        """
        Adds a conversational message attributed to one of the three entities:
        - You (user/mic/manual input)
        - Third-Party (loopback/system audio)
        - AI (assistant)
        """
        ent = self._normalize_entity(entity)
        if ent == "assistant":
            self.add_assistant_message(content)
            return

        msg = {"role": "user", "content": content}
        if ent == "third_party":
            if speaker_id:
                safe_id = self._safe_openai_name_component(str(speaker_id), max_len=24)
                safe_label = self._safe_openai_name_component(str(speaker_label), max_len=28) if speaker_label else ""
                if safe_label and safe_id:
                    msg["name"] = f"third_party_{safe_label}__{safe_id}"
                elif safe_id:
                    msg["name"] = f"third_party_{safe_id}"
                else:
                    msg["name"] = "third_party"
            else:
                msg["name"] = "third_party"
        else:
            msg["name"] = "you"
        self.history.append(msg)

    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content, "name": "you"})

    def add_assistant_message(self, content):
        self.history.append({"role": "assistant", "content": content})

    def format_recent_history(self, limit: int | None = None) -> str:
        messages = self.history[-int(limit):] if limit is not None else self.history
        lines = []
        for m in messages:
            label = self._label_from_message(m)
            lines.append(f"{label}: {m.get('content', '')}")
        return "\n".join(lines)

    async def stream_reply(self, *, extra_messages=None, system_prompt=None, add_to_history: bool = True):
        """
        Streams a reply from the LLM based on current history.
        Yields chunks of text.
        """
        if extra_messages is None:
            extra_messages = []
        prompt = self.system_prompt if system_prompt is None else system_prompt
        messages = [{"role": "system", "content": prompt}] + self.history + list(extra_messages)

        try:
            stream = await self.chat_create(
                messages=messages,
                stream=True,
            )
            
            full_response = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
            
            # Save the full response to history for context
            if add_to_history and full_response:
                self.add_assistant_message(full_response)
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            yield f"\n[Error: {str(e)}]"

    def clear_history(self):
        self.history = []
