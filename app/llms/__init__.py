from .anthropic import Anthropic
from .deepseek import DeepSeek
from .google import GeminiFamily
from .lgai import LGAI
from .llava import LlavaModels
from .meta import MetaLlama
from .openai import OpenAI, OpenAIOSS
from .qwen import QwenModels
from .xai import XAI

llm_models = [
    Anthropic(),
    DeepSeek(),
    GeminiFamily(),
    LGAI(),
    LlavaModels(),
    MetaLlama(),
    OpenAI(), OpenAIOSS(),
    QwenModels(),
    XAI()
]

__all__ = [
    'llm_models',
    # ------- Individual LLM schemas -------
    'Anthropic',
    'DeepSeek',
    'GeminiFamily',
    'LGAI',
    'LlavaModels',
    'MetaLlama',
    'OpenAI',
    'OpenAIOSS',
    'QwenModels',
    'XAI',
]