from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
from fastapi.exceptions import RequestValidationError
from app.api.errors import global_exception_handler, http_exception_handler, validation_exception_handler
from app.api.routes import chat

app = FastAPI(title="TU BS KI-Toolbox API Wrapper", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(httpx.HTTPStatusError, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(chat.router, prefix="/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
