from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1", model="openai/gpt-3.5-turbo", default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
        self.model = model
        self.history = []
        self.system_prompt = "You are a helpful AI assistant running on the user's desktop. Keep your answers concise and helpful."

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
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
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
