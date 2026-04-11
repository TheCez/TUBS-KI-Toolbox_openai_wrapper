"""
/v1/models endpoint.
Returns all available TU-BS models (cloud + local) and Anthropic-mapped aliases.
"""

import time
from fastapi import APIRouter
from app.models.tubs import CloudModel, LocalModel
from app.services.model_map import get_anthropic_model_map

router = APIRouter()


@router.get("/models")
async def list_models():
    """Lists all available models in OpenAI-compatible format."""
    current_time = int(time.time())
    models_data = []

    # TU-BS Cloud Models
    for model in CloudModel:
        models_data.append({
            "id": model.value,
            "object": "model",
            "created": current_time,
            "owned_by": "tu-bs",
        })

    # TU-BS Local (On-Premise) Models
    for model in LocalModel:
        models_data.append({
            "id": model.value,
            "object": "model",
            "created": current_time,
            "owned_by": "tu-bs",
        })

    # Anthropic mapped aliases
    for anthropic_id in get_anthropic_model_map():
        models_data.append({
            "id": anthropic_id,
            "object": "model",
            "created": current_time,
            "owned_by": "anthropic-shim",
        })

    return {"object": "list", "data": models_data}
