"""Server-side Gemini chat for the reconstruction studio."""
from __future__ import annotations

import json
import re
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StringConstraints

from ..auth import get_current_user
from ..config import gemini_settings
from ..db import User
from ..ratelimit import RateLimiter, enforce
from ..schemas import ApiError, ok

router = APIRouter(prefix="/chat", tags=["chat"])
chat_limiter = RateLimiter(8, 60)
SYSTEM_PROMPT = (
    "You are the study assistant for 3DFUN AI, a 3D computer vision course project. "
    "Help students choose reference images, understand single-view reconstruction, "
    "compare TripoSR with the course silhouette pipeline using Depth Anything V2 or MiDaS, and use mesh exports. "
    "Explain ideas concisely in plain text. You cannot see uploaded images or run jobs."
)


class Message(BaseModel):
    role: Literal["user", "model"]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=11)


@router.get("/status")
def chat_status(_user: User = Depends(get_current_user)):
    key, model = gemini_settings()
    return ok({"configured": bool(key), "model": model})


@router.post("")
def chat(payload: ChatRequest, user: User = Depends(get_current_user)):
    key, model = gemini_settings()
    if not key:
        raise ApiError("CHAT_NOT_CONFIGURED", "Set GEMINI_API_KEY in the local .env file.", status=503)
    if not re.fullmatch(r"gemini-[a-z0-9.-]+", model):
        raise ApiError("CHAT_MODEL_INVALID", "Set GEMINI_MODEL to a valid Gemini model ID.", status=503)
    if len(payload.messages) % 2 != 1 or any(
        item.role != ("model" if i % 2 else "user") for i, item in enumerate(payload.messages)
    ):
        raise ApiError("CHAT_HISTORY_INVALID", "Chat history must alternate and end with your message.", status=422)
    enforce(chat_limiter, user.id, "chat")
    data = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": item.role, "parts": [{"text": item.text}]} for item in payload.messages],
        "generationConfig": {"maxOutputTokens": 1200},
    }
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urlopen(request, timeout=35) as response:
            result = json.load(response)
    except HTTPError as exc:
        exc.close()
        if exc.code in (400, 401, 403):
            raise ApiError("CHAT_CREDENTIALS", "Check your Gemini key and model access in Google AI Studio.", status=502) from None
        if exc.code == 404:
            raise ApiError("CHAT_MODEL_UNAVAILABLE", "The configured Gemini model is unavailable. Check GEMINI_MODEL.", status=502) from None
        if exc.code == 429:
            raise ApiError("CHAT_QUOTA", "Gemini quota reached. Try again later or check your API limits.", status=429, retryable=True) from None
        raise ApiError("CHAT_UPSTREAM", "Gemini is temporarily unavailable. Try again shortly.", status=502, retryable=True) from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise ApiError("CHAT_UPSTREAM", "Could not reach Gemini. Check your internet connection and try again.", status=502, retryable=True) from None
    parts = next(iter(result.get("candidates", [])), {}).get("content", {}).get("parts", [])
    reply = "\n".join(part["text"] for part in parts if part.get("text") and not part.get("thought")).strip()
    if not reply:
        raise ApiError("CHAT_EMPTY_REPLY", "Gemini returned no reply. Try rephrasing your question.", status=502)
    return ok({"text": reply, "model": model})
