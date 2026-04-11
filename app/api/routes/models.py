import os
import json
import time
from fastapi import APIRouter
from app.models.openai import CloudModel, LocalModel

router = APIRouter()

def get_anthropic_model_map():
    env_map = os.getenv("ANTHROPIC_MODEL_MAP")
    if env_map:
        try:
            return json.loads(env_map)
        except Exception:
            pass
            
    # Default fallbacks targeting our known Tu-BS models
    return {
        "claude-3-5-opus-latest": CloudModel.GPT_5_4.value,
        "claude-3-opus-20240229": CloudModel.GPT_5_4.value,
        "claude-3-5-sonnet-latest": CloudModel.GPT_O3.value,
        "claude-3-5-sonnet-20241022": CloudModel.GPT_4O.value,
        "claude-3-5-sonnet-20240620": CloudModel.GPT_4O.value,
        "claude-3-7-sonnet-20250219": CloudModel.GPT_O3.value,
        "claude-3-5-haiku-20241022": CloudModel.GPT_O4_MINI.value,
        "claude-3-haiku-20240307": CloudModel.GPT_4O_MINI.value,
    }

@router.get("/models")
async def list_models():
    models_data = []
    current_time = int(time.time())
    
    # 1. Add all direct Tubs Cloud Models
    for model in CloudModel:
        models_data.append({
            "id": model.value,
            "object": "model",
            "created": current_time,
            "owned_by": "tu-bs"
        })
        
    # 2. Add all direct Tubs Local Models
    for model in LocalModel:
        models_data.append({
            "id": model.value,
            "object": "model",
            "created": current_time,
            "owned_by": "tu-bs"
        })
        
    # 3. Add Anthropic mapped mock models so Anthropic frontends spot them
    anthropic_map = get_anthropic_model_map()
    for anthropic_model, mapped_target in anthropic_map.items():
        models_data.append({
            "id": anthropic_model,
            "object": "model",
            "created": current_time,
            "owned_by": "anthropic-shim"
        })
        
    return {
        "object": "list",
        "data": models_data
    }
