from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.models.openai import ErrorResponse, ErrorDetail
import httpx
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=exc.detail,
                    type="api_error",
                    code=str(exc.status_code)
                )
            ).model_dump()
        )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc),
                type="server_error",
                code="internal_error"
            )
        ).model_dump()
    )

async def http_exception_handler(request: Request, exc: httpx.HTTPStatusError):
    return JSONResponse(
        status_code=exc.response.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.response.text,
                type="api_error",
                code=f"http_{exc.response.status_code}"
            )
        ).model_dump()
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc.errors()),
                type="invalid_request_error",
                code="validation_error"
            )
        ).model_dump()
    )
