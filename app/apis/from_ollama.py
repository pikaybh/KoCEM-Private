import os
import json
import requests

from models import APIBase
from utils.logs import set_logger


OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_COOKIE = os.getenv("OLLAMA_COOKIE", "")

logger = set_logger(__name__)



class OllamaAPI(APIBase):
    """
    Remote API class for initializing and managing chat models via a remote endpoint.
    Inherits from APIBase and provides additional functionality.
    """
    def __init__(self, 
        model_id: str, 
        endpoint: str = OLLAMA_ENDPOINT,
        base_url: str | None = None,
        cookie: str = OLLAMA_COOKIE,
        cookie_key: str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        # Normalize model id to Ollama tag format: always replace only the last '-' with ':' if present
        if "-" in model_id:
            _idx = model_id.rfind("-")
            tag = model_id[:_idx] + ":" + model_id[_idx + 1:]
        else:
            tag = model_id
        self._model_tag = tag.split("/")[-1]
        self._base_url = (base_url or endpoint).rstrip("/")
        self._cookie = cookie
        self._cookie_key = cookie_key

    def _build_cookie_header(self) -> dict:
        if not self._cookie:
            return {}
        c = self._cookie.strip()
        # try:
        #     if c.startswith("{") and c.endswith("}"):
        #         obj = json.loads(c)
        #         if isinstance(obj, dict) and obj:
        #             return {"Cookie": "; ".join(f"{k}={v}" for k, v in obj.items())}
        # except Exception:
        #     logger.warning("OLLAMA cookie JSON parse failed; using raw value.")
        if "=" in c:
            return {"Cookie": c}
        if self._cookie_key:
            return {"Cookie": f"{self._cookie_key}={c}"}
        logger.warning("OLLAMA cookie has no key; sending raw value which may be rejected.")
        return {"Cookie": c}

    def _to_ollama_messages(self, msgs):
        def _content_to_text(content) -> str:
            # Ollama expects a string; strip non-text parts (e.g., images) if present
            try:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, str):
                            texts.append(part)
                        elif isinstance(part, dict):
                            if "text" in part and isinstance(part["text"], str):
                                texts.append(part["text"])
                            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                                texts.append(part.get("text", ""))
                            elif isinstance(part.get("content"), str):
                                texts.append(part["content"]) 
                            # ignore non-text (e.g., image_url)
                    return "\n".join(t for t in texts if t).strip()
                if isinstance(content, dict):
                    if "text" in content and isinstance(content["text"], str):
                        return content["text"]
                    if isinstance(content.get("content"), str):
                        return content["content"]
                return str(content)
            except Exception:
                return str(content)

        out = []
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and "role" in m:
                    text = _content_to_text(m.get("content", ""))
                    out.append({"role": m["role"], "content": text})
                else:
                    name = m.__class__.__name__.lower()
                    role = "system" if "system" in name else ("assistant" if "ai" in name else "user")
                    content = getattr(m, "content", str(m))
                    out.append({"role": role, "content": _content_to_text(content)})
        else:
            out.append({"role": "user", "content": _content_to_text(msgs)})
        return out

    def _post_chat(self, messages, timeout: int | None):
        url = f"{self._base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        headers.update(self._build_cookie_header())
        payload = {
            "model": self._model_tag,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
        }
        logger.debug(f"Ollama request headers: {headers}")
        resp = requests.post(url, json=payload, headers=headers, timeout=(10, timeout or 60))
        if resp.status_code == 401:
            raise requests.HTTPError("Not Authenticated - INVALIDCOOKIE", response=resp)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content") or data.get("response") or str(data)
        class _Resp:
            def __init__(self, c, r):
                self.content = c
                self._r = r
            def dict(self):
                return self._r
        return _Resp(content, data)

    def _invoke_with_retry(self, prompt_msgs, max_retries: int, max_timeout: int):
        import time
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._post_chat(prompt_msgs, timeout=max_timeout if (max_timeout and max_timeout > 0) else None)
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status and 400 <= status < 500 and status != 429:
                    last_err = e
                    logger.error(f"Ollama HTTP error (no retry): {e}, status={status}, body={getattr(e.response,'text','')}")
                    raise
                last_err = e
            except (requests.Timeout, requests.ConnectionError, Exception) as e:
                last_err = e
            if attempt == max_retries:
                logger.error(f"Model invoke failed after {attempt} attempts: {last_err}")
                raise last_err
            backoff = min(2 ** (attempt - 1), 10)
            logger.warning(f"Invoke error (attempt {attempt}/{max_retries}): {last_err}. Retrying in {backoff}s...")
            time.sleep(backoff)



__all__ = ["OllamaAPI"]
