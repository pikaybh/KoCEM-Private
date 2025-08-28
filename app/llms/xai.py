from schemas import (
    LanguageModel,
    ModelSize,
    ModelVersion,
    Modality,
)
from models import LLMBase



class XAI(LLMBase):
    def __init__(self, 
        provider: str = "xai", 
        docs: str = "https://huggingface.co/xai-org/grok-1"
    ):
        super().__init__(provider, docs)

    models = [
        LanguageModel(
            name="jmorgan/grok",
            description="A class-leading natively multimodal model that offers superior text and visual intelligence.",
            version=ModelVersion(
                releases=["jmorgan/grok-latest"],
                stable="jmorgan/grok-latest"
            ),
            size=ModelSize(parameters=314_000_000, aunounced=True),
            modality=Modality(
                input_type=["text"],
                output_type=["text"]
            ),
            features={
                "context_window": 128_000,
                "mixture_of_experts": 16,
            },
            pricing="open-source"
        )
    ]



__all__ = ['XAI']