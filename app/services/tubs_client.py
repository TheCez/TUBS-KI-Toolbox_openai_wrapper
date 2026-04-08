import httpx
import json
from typing import AsyncGenerator
from fastapi import HTTPException
from app.models.tubs import is_local_model

TUBS_CLOUD_URL = "https://ki-toolbox.tu-braunschweig.de/api/v1/chat/send"
TUBS_LOCAL_URL = "https://ki-toolbox.tu-braunschweig.de/api/v1/localChat/send"

async def _non_stream_response(client: httpx.AsyncClient, url: str, headers: dict, req_kwargs: dict):
    try:
        response = await client.post(url, headers=headers, **req_kwargs)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # KI-Toolbox API streams NDJSON by default, we capture it all and wait for "done" chunk
        final_data = {}
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if chunk.get("type") == "done":
                    final_data = chunk
                    break
        return final_data
    finally:
        await client.aclose()

async def _stream_response(client: httpx.AsyncClient, url: str, headers: dict, req_kwargs: dict):
    try:
        async with client.stream("POST", url, headers=headers, **req_kwargs) as response:
            if response.status_code != 200:
                await response.aread()
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)
    finally:
        await client.aclose()

async def async_send_tubs_request(
    payload: dict,
    images: list,
    bearer_token: str,
    stream: bool
):
    url = TUBS_LOCAL_URL if is_local_model(payload.get("model", "")) else TUBS_CLOUD_URL
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }
    
    req_kwargs = {}
    if images:
        files = []
        for img in images:
            fname, fbytes, ftype = img
            files.append(("chatAttachment", (fname, fbytes, ftype)))
        
        data = {
            "jsonBody": json.dumps(payload)
        }
        req_kwargs["data"] = data
        req_kwargs["files"] = files
    else:
        headers["Content-Type"] = "application/json"
        req_kwargs["json"] = payload
        
    client = httpx.AsyncClient(timeout=60.0)
    
    if stream:
        return _stream_response(client, url, headers, req_kwargs)
    else:
        return await _non_stream_response(client, url, headers, req_kwargs)
