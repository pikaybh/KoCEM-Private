import os, requests

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from models.prompt import PromptManager
from utils.logs import set_logger


logger = set_logger(__name__)



class _SimpleResponse:
    """Lightweight response wrapper to fit APIBase expectations."""
    def __init__(self, content: str, raw: dict):
        self.content = content
        self._raw = raw

    def dict(self):
        return self._raw
    


class JudgeResponse(BaseModel):
    label: str = Field(..., description="The label of the answer, e.g., A, B, C, D")
    reason: str = Field(..., description="The reason for the judgment")



def check_label(query: str, answer: str) -> str:
    """
    Check the label of the answer based on the provided query and answer text.

    Args:
        query (str): The question text containing multiple-choice options.
        answer (str): The assistant's answer text.

    Returns:
        str: The determined label (A, B, C, D) or "Unknown" if no match is found.
    """
    assistant = init_chat_model("openai:gpt-4.1").with_structured_output(JudgeResponse)
    prompt_temp = PromptManager("check-label", locale="en")
    # Use templating so {query} and {answer} are injected at invoke time
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_temp.system),
        ("human", prompt_temp.human.format(
            query="{query}", 
            answer="{answer}"
        )),
    ])
    chain = prompt | assistant

    logger.debug(f"{prompt = }")
    logger.debug(f"{query = }")
    logger.debug(f"{answer = }")
    
    response = chain.invoke({"query": query, "answer": answer})
    logger.debug(f"Judge Response: {response}")
    return response


def check_label_w_gpt_5(query: str, answer: str) -> str:
    """
    Check the label of the answer based on the provided query and answer text.

    Args:
        query (str): The question text containing multiple-choice options.
        answer (str): The assistant's answer text.

    Returns:
        str: The determined label (A, B, C, D) or "Unknown" if no match is found.
    """

    prompt = PromptManager("check-label", locale="en")
    # Use templating so {query} and {answer} are injected at invoke time

    logger.debug(f"{prompt = }")
    logger.debug(f"{query = }")
    logger.debug(f"{answer = }")
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
        },
        json={
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.human.format(query=query, answer=answer)},
            ],
        },
        timeout = 60 * os.getenv("TIMEOUT", "5"),
        text_format = JudgeResponse
    )
    response.raise_for_status()
    data = response.json()
    
    logger.debug(f"Judge Response: {data}")

    content = None
    try:
        # OpenAI style: choices[0].message.content
        content = data["choices"][0]["message"]["content"]
    except Exception:
        # fallback to raw string
        content = str(data)
    return _SimpleResponse(content=content, raw=data)


__all__ = [
    "check_label",
    "check_label_w_gpt_5"
]