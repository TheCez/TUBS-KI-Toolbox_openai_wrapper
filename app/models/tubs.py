from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel

class CloudModel(str, Enum):
    GPT_5_4 = "gpt-5.4"
    GPT_5_2 = "gpt-5.2"
    GPT_5_1 = "gpt-5.1"
    GPT_5 = "gpt-5"
    GPT_O3 = "o3"
    GPT_4_1 = "gpt-4.1"
    GPT_O4_MINI = "o4-mini"
    GPT_O3_MINI = "o3-mini"
    GPT_O1 = "o1"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"

class LocalModel(str, Enum):
    GPT_OSS_120B = "OpenAI/GPT-OSS-120B"
    QWEN_3_30B = "Qwen/Qwen3-30B-A3B"
    QWEN_2_5_CODER = "Qwen/Qwen2.5-Coder-32B-Instruct"
    MS_PHI_4 = "Microsoft/Phi-4"
    MISTRAL_SMALL_24B = "mistralai/Mistral-Small-24B-Instruct-2501"
    MAGISTRAL_SMALL = "mistralai/Magistral-Small-2509"

def is_local_model(model_id: str) -> bool:
    try:
        LocalModel(model_id)
        return True
    except ValueError:
        return False

def is_cloud_model(model_id: str) -> bool:
    try:
        CloudModel(model_id)
        return True
    except ValueError:
        return False

class TubsChatRequest(BaseModel):
    thread: Optional[str] = None # Expects null or thread string id
    prompt: str
    model: str
    customInstructions: Optional[str] = None
    hideCustomInstructions: Optional[bool] = False

# Note: KI-Toolbox API responses
# Non-stream chunk type: "done"
# Stream chunk type: "chunk"
# {"type": "done", "response": "...", "promptTokens": 1, "responseTokens": 2, "totalTokens": 3, "thread": {"id": "..."}}
