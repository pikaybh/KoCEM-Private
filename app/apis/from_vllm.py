import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.llms import VLLMOpenAI

from models import APIBase
from utils.llm import get_provider


load_dotenv()
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT")



class VLLMAPI(APIBase):
    def __init__(self, 
        model_id: str, 
        base_url: str = VLLM_ENDPOINT, 
        api_key: str = "EMPTY",
        **kwargs
    ):
        super().__init__(**kwargs)
        provider = get_provider(model_id)
        self.model_id = model_id
        self.model = VLLMOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY", api_key),
            openai_api_base=base_url if "/v1" in base_url else f"{base_url}/v1",
            model_name=f"{provider}/{model_id}",
            model_kwargs={"stop": ["."]},
        )


__all__ = ["VLLMAPI"]