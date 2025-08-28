from llms import llm_models

from .standalone import GPUFreeAPI
from .from_langserve import RemoteAPI
from .from_vllm import VLLMAPI
from .from_ollama import OllamaAPI


module = {
    "anthropic": GPUFreeAPI,
    "google": GPUFreeAPI,
    "openai": GPUFreeAPI,
    'gpt-oss': OllamaAPI,
    "Qwen": VLLMAPI,
    "meta": OllamaAPI,
    "liuhaotian": OllamaAPI,
    "deepseek": OllamaAPI,
    "xai": OllamaAPI,
}

for kprovider, vapi in module.items():
    for llm in llm_models:
        if kprovider == llm.provider:
            module[kprovider] = vapi
            break


__all__ = ["module"]
