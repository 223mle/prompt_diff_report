from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class Messages(BaseModel):
    messages: list[Message]


class GenerateParameters(BaseModel):
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=256, le=16384)

    model_config = {
        'extra': 'allow',  # provider 固有オプションを許可
    }
