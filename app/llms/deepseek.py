from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
)
from models import LLMBase



class DeepSeek(LLMBase):
    def __init__(self, 
        provider: str = "deepseek", 
        docs: str = "https://ollama.com/library/deepseek-r1"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="deepseek-r1:671b",
            description="A class-leading natively multimodal model that offers superior text and visual intelligence.",
            version=ModelVersion(
                releases=["deepseek-r1-671b"],
                stable="deepseek-r1-671b"
            ),
            size=ModelSize(parameters=671_000_000, aunounced=True),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 128_000,
                "mixture_of_experts": 16,
            },
            pricing="open-source"
        ),
        LanguageModel(
            name="deepseek-r1:8b",
            description="A class-leading natively multimodal model that offers superior text and visual intelligence.",
            version=ModelVersion(
                releases=["deepseek-r1-8b"],
                stable="deepseek-r1-8b"
            ),
            size=ModelSize(parameters=8_000_000, aunounced=True),
            modality=Modality(
                input_type=["text", "image"],
                output_type=["text"]
            ),
            features={
                "context_window": 128_000,
                "mixture_of_experts": 16,
            },
            pricing="open-source"
        )
    ]



__all__ = ['DeepSeek']