import os

from langchain_community.llms import VLLMOpenAI
from datasets import load_dataset
from tqdm import tqdm

from models import APIBase
from utils.llm import get_provider
from utils.logs import set_logger
from utils.data import save_json
from utils.ds import call_features
from utils.eval import evaluate, evaluate_difficulties


logger = set_logger(__name__)
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT")
VLLM_COOKIE = os.getenv("VLLM_COOKIE", "")




import requests
import json


class _SimpleResponse:
    """Lightweight response wrapper to fit APIBase expectations."""
    def __init__(self, content: str, raw: dict):
        self.content = content
        self._raw = raw

    def dict(self):
        return self._raw


class VLLMAPI(APIBase):
    def __init__(self, 
        model_id: str, 
        base_url: str = VLLM_ENDPOINT, 
        cookie: str = VLLM_COOKIE,
        cookie_key: str = None,
        api_key: str = "EMPTY",
        **kwargs
    ):
        super().__init__(**kwargs)
        # provider = get_provider(model_id)
        self.model_id = model_id
        self.base_url = base_url
        self.cookie = cookie
        self.cookie_key = cookie_key

    def _send_curl_request(self, prompt_msgs, timeout: int | None = None):
        """
        Send POST to base_url[/v1/chat/completions] with {model, messages} body and Cookie header.
        """
        # Build endpoint
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1/chat/completions"):
            endpoint = base
        elif base.endswith("/v1"):
            endpoint = f"{base}/chat/completions"
        else:
            endpoint = f"{base}/v1/chat/completions"

        # Convert prompt to OpenAI-compatible messages array
        def to_message_list(msgs):
            if isinstance(msgs, list):
                out = []
                for m in msgs:
                    # Already dict with role/content
                    if isinstance(m, dict) and "role" in m and "content" in m:
                        out.append({"role": m["role"], "content": m["content"]})
                        continue
                    # LangChain message objects
                    role = getattr(m, "type", None)
                    if not role:
                        name = m.__class__.__name__.lower()
                        role = "system" if "system" in name else "user"
                    content = getattr(m, "content", m)
                    out.append({"role": role, "content": content})
                return out
            # String fallback
            return [{"role": "user", "content": str(msgs)}]

        messages = to_message_list(prompt_msgs)

        headers = {"Content-Type": "application/json"}
        # 쿠키 처리 개선: 여러 개 지원 (세미콜론 연결 또는 JSON 문자열)
        if self.cookie:
            cookie_val = self.cookie.strip()
            # 1) JSON 형태로 전달된 경우: {"k1":"v1", "k2":"v2"}
            if (cookie_val.startswith("{") and cookie_val.endswith("}")):
                try:
                    cookie_obj = json.loads(cookie_val)
                    if isinstance(cookie_obj, dict) and cookie_obj:
                        headers["Cookie"] = "; ".join([f"{k}={v}" for k, v in cookie_obj.items()])
                    else:
                        logger.warning("쿠키 JSON이 비어있거나 객체가 아닙니다. 원문을 Cookie 헤더로 사용합니다.")
                        headers["Cookie"] = cookie_val
                except Exception:
                    logger.warning("쿠키 JSON 파싱 실패. 원문을 Cookie 헤더로 사용합니다.")
                    headers["Cookie"] = cookie_val
            # 2) key=value 세미콜론 연결 문자열 전체가 온 경우 그대로 사용 (cookie_key 무시)
            elif "=" in cookie_val:
                headers["Cookie"] = cookie_val
            # 3) value만 온 경우 cookie_key가 있으면 결합
            elif self.cookie_key:
                headers["Cookie"] = f"{self.cookie_key}={cookie_val}"
            # 4) 아무 키 정보가 없으면 경고
            else:
                logger.warning("쿠키 값이 key=value 형식이 아닙니다. 서버에서 인증이 거부될 수 있습니다. (예: cookie_key=your_cookie_value)")
                headers["Cookie"] = cookie_val
        logger.debug(f"Outgoing headers: {headers}")
        payload = {"model": self.model_id, "messages": messages}
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout if (timeout and timeout > 0) else 60)
        response.raise_for_status()
        data = response.json()
        # OpenAI/vLLM compatible extraction
        content = None
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: try top-level keys common in some proxies
            content = data.get("content") or data.get("text") or str(data)
        return _SimpleResponse(content=content, raw=data)

    def _invoke_with_retry(self, prompt_msgs, max_retries: int, max_timeout: int):
        """Send messages via HTTP POST to /v1/chat/completions with retry and per-call timeout."""
        import time
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._send_curl_request(prompt_msgs, timeout=max_timeout if (max_timeout and max_timeout > 0) else None)
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                # Don't retry most client errors (except 429)
                if status and 400 <= status < 500 and status != 429:
                    last_err = e
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


__all__ = ["VLLMAPI"]