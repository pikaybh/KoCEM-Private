import os, requests, time
from langchain.chat_models import init_chat_model

from models import APIBase
from utils.logs import set_logger


logger = set_logger(__name__)



class _SimpleResponse:
    """Lightweight response wrapper to fit APIBase expectations."""
    def __init__(self, content: str, raw: dict):
        self.content = content
        self._raw = raw

    def dict(self):
        return self._raw



class _OpenAIHTTPChatInvoker:
    """HTTP-based invoker that mimics LangChain's .invoke interface for chat."""
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1/chat/completions"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def invoke(self, messages, timeout: int | tuple | None = None):
        # Configure timeout: (connect, read)
        if timeout is None:
            req_timeout = (10, 300)
        elif isinstance(timeout, (int, float)):
            req_timeout = (10, float(timeout))
        elif isinstance(timeout, tuple) and len(timeout) == 2:
            req_timeout = timeout
        else:
            req_timeout = (10, 60)

        resp = requests.post(
            self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": messages,
            },
            timeout=req_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"{data = }")

        content = None
        try:
            # OpenAI style: choices[0].message.content
            content = data["choices"][0]["message"]["content"]
        except Exception:
            # fallback to raw string
            content = str(data)
        return _SimpleResponse(content=content, raw=data)



class GPUFreeAPI(APIBase):
    """
    Standard API class for initializing and managing chat models.
    Inherits from APIBase and provides additional functionality.
    """
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        if "gpt-5" in model_id:
            self._model = _OpenAIHTTPChatInvoker(
                model="gpt-5",
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )
        else:
            self._model = init_chat_model(model_id.replace("/", ":"))

    # Override retry to avoid outer thread timeout fighting HTTP timeout
    def _invoke_with_retry(self, prompt_msgs, max_retries: int, max_timeout: int):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                # Adaptive per-attempt read timeout (cap at 600s)
                base = max_timeout if (max_timeout and max_timeout > 0) else 60
                attempt_timeout = min(int(base * (2 ** (attempt - 1))), 600)
                logger.debug(f"Attempt {attempt}/{max_retries}: using read timeout {attempt_timeout}s")
                return self.model.invoke(prompt_msgs, timeout=(10, attempt_timeout))
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                # Do not retry most 4xx except 429
                if status and 400 <= status < 500 and status != 429:
                    last_err = e
                    raise
                last_err = e
            except (requests.Timeout, requests.ReadTimeout, requests.ConnectionError, Exception) as e:
                last_err = e
            if attempt == max_retries:
                logger.error(f"Model invoke failed after {attempt} attempts: {last_err}")
                raise last_err
            backoff = min(2 ** (attempt - 1), 10)
            logger.warning(f"Invoke error (attempt {attempt}/{max_retries}): {last_err}. Retrying in {backoff}s...")
            time.sleep(backoff)
        return "The model failed to answer."



__all__ = ["GPUFreeAPI"]